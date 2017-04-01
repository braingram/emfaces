#!/usr/bin/env python

import os
import pickle
import sys

from PIL import Image
import requests
import numpy as np
from StringIO import StringIO

import cv2
import slackclient


# TODO fill these in
base_url = "http://base of your catmaid tilestore"
project_id = None
stack_id = None
server_url = None


url_fmt = "{base_url}/{zoom}/{z}/{row}/{col}.jpg"
zmin = 4
zmax = 1920

cfn = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"

scaleFactor = 1.1
minNeighbors = 6
minSize = (30, 30)
default_cascade = cv2.CascadeClassifier(cfn)

outdir = 'faces'
if not os.path.exists(outdir):
    os.makedirs(outdir)

state = {}


def get_url(row=0, col=0, z=0, zoom=1):
    url = url_fmt.format(
        row=row, col=col, zoom=zoom, z=z, base_url=base_url)
    print("fetching: %s" % url)
    response = requests.get(url)
    if response.status_code != 200:
        return None
    im = np.array(Image.open(StringIO(response.content)))
    return im


def face_detect(image, cascade=default_cascade):
    if image.ndim > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Detect faces in the image
    if hasattr(cascade, 'detectMultiScale3'):
        faces, reject_levels, weights = cascade.detectMultiScale3(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            #outputRejectLevels=True,
            #flags = cv2.CV_HAAR_SCALE_IMAGEa
        )
    else:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            #outputRejectLevels=True,
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        weights = [1 for _ in faces]
        reject_levels = [None for _ in faces]

    if len(faces):
        print("Found %i faces" % len(faces))
        for (f, r, w) in zip(faces, reject_levels, weights):
            print("%0.1f, %s, %s" % (w, r, f))
    else:
        print("No faces found")
    return faces, weights


def draw_faces(image, faces, weights, show=True):
    # Draw a rectangle around the faces
    font = cv2.FONT_HERSHEY_SIMPLEX
    if image.ndim > 2:
        bc = (0, 255, 0)
        tc = (255, 0, 0)
    else:
        bc = 0
        tc = 0
    for ((x, y, w, h), weight) in zip(faces, weights):
        print(x, y, w, h, weight)
        cv2.rectangle(image, (x, y), (x+w, y+h), bc, 2)
        cv2.putText(
            image, '%0.2f' % weight, (x+w, y+h),
            font, 1, tc, 2, cv2.LINE_AA)

    if show:
        cv2.imshow("Faces found", image)
        while cv2.waitKey(0) != ord('q'):
            pass


def crop_out_faces(image, faces, border=0.1):
    crops = []
    ih, iw = image.shape
    for f in faces:
        x, y, w, h = f
        m = int(border * w)
        # range check
        l = max(0, x-m)
        r = min(iw, x+h+m)
        t = max(0, y-m)
        b = min(ih, y+h+m)
        cim = image[t:b, l:r]
        crops.append(cim)
    return crops


def show_faces(crops, faces, weights, delay=3000):
    for (i, (f, c, w)) in enumerate(zip(faces, crops, weights)):
        cv2.imshow("%0.1f[%i]: %s" % (w, i, f), c)
        cv2.waitKey(delay)
    cv2.destroyAllWindows()


#def crawl(z=1, row=0, col=0, delay=1000):
def crawl(z=1060, row=20, col=20, delay=1000):
    kwargs = {'z': z, 'row': row, 'col': col}
    im = get_url(**kwargs)
    while im is None:
        kwargs['z'] += 1
        im = get_url(**kwargs)
    while im is not None:
        while im is not None:
            while im is not None:
                # process image
                fcs, ws = face_detect(im)
                if len(fcs):
                    yield fcs, im, kwargs['z'], kwargs['row'], kwargs['col']
                    # crops = crop_out_faces(im, fcs)
                    # show_faces(crops, fcs, ws, delay)
                # increment row
                kwargs['row'] += 1
                im = get_url(**kwargs)
            kwargs['row'] = row
            kwargs['col'] += 1
            im = get_url(**kwargs)
        kwargs['col'] = col
        kwargs['z'] += 1
        im = get_url(**kwargs)


def single():
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['row'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kwargs['col'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kwargs['z'] = int(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs['zoom'] = int(sys.argv[4])
    im = get_url(**kwargs)
    if im is None:
        sys.exit(1)
    fcs, ws = face_detect(im)
    if not len(fcs):
        print("No faces found")
        sys.exit(0)
    crops = crop_out_faces(im, fcs)
    show_faces(crops, fcs, ws)
    #draw_faces(im, fcs, ws)


def resume(sfn=None):
    if sfn is None:
        sfn = 'state.p'
    # read in state
    if os.path.exists(sfn):
        with open(sfn, 'r') as f:
            state = pickle.load(f)
    else:
        state = {
            'fns': [], 'crawl': {}, 'index': 0,
            'visited': []}

    # check for previous 'found' image
    if len(state['fns']) == 0:
        # crawl for more
        if 'row' in state['crawl']:
            state['crawl']['row'] += 1
        ci = crawl(**state['crawl'])
        fcs, im, z, row, col = ci.next()
        print("Found %i faces in: %i, %i, %i" % (len(fcs), z, row, col))
        state['crawl'] = {'z': z, 'row': row, 'col': col}

        # write out faces to files
        crops = crop_out_faces(im, fcs)

        # save crops
        fns = []
        for c in crops:
            fn = '%s/%05i.jpg' % (outdir, state['index'])
            state['index'] += 1
            cv2.imwrite(fn, c)
            fns.append(fn)
        state['fns'] = fns
        state['fcs'] = list(fcs)
    imfn = state['fns'].pop()
    fc = state['fcs'].pop()
    state['face'] = fc

    # save state
    with open(sfn, 'w') as f:
        pickle.dump(state, f)

    return imfn, state


def upload_file(fn, state):
    fx, fy, fw, fh = state['face']
    dx = fx + fw / 2
    dy = fy + fh / 2
    url = '{server}/?pid={p}&zp={z}' \
        '&yp={y}&xp={x}&tool=navigator&sid0={s}&s0=1'.format(
            server=server_url,
            z=state['crawl']['z'] * 40,
            x=(state['crawl']['col'] * 1024 + dx) * 8,
            y=(state['crawl']['row'] * 1024 + dy) * 8,
            p=project_id,
            s=stack_id)
    print url

    if 'SLACK_TOKEN' not in os.environ:
        raise RuntimeError("Missing environment variable SLACK_TOKEN")

    c = slackclient.SlackClient(os.environ['SLACK_TOKEN'])
    c.api_call(
        'files.upload', channels='#random',
        filename=fn, file=open(fn, 'rb'), initial_comment=url)

if __name__ == '__main__':
    # single()
    # crawl()
    fn, state = resume()
    lfn = 'face.jpg'
    if os.path.exists(lfn):
        os.remove(lfn)
    os.symlink(fn, lfn)
    print(fn)
    # post to slack
    upload_file(fn, state)
