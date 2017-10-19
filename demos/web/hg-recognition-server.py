#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import openface

import json
import time
import gzip

from gesture_recognizer import GestureRecognizer
gs = GestureRecognizer.load_model(name = "sign_detector.pkl.gz")

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# For TLS connections
# tls_crt = os.path.join(fileDir, 'tls', 'localhost.crt')
# tls_key = os.path.join(fileDir, 'tls', 'localhost.key')

tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training
        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

    def processFrame(self, dataURL, identity):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)
        img.save('/Users/xinfangyuan/Ai/openface/demos/web/hg/myphoto.jpg', 'JPEG')
        print('save succeed')
        # image = cv2.imread('/Users/xinfangyuan/Ai/openface/demos/web/hg/myphoto.jpg')
        # pos, ped = gs.recognize_gesture(image)
        # print 'ok to gs'
        # print pos
        # print ped

        content = 'data:image/png;base64,' + \
        urllib.quote(base64.b64encode(imgF.buf))
        msg = {
            "type": "ANNOTATED",
            "content": content
        }
        self.sendMessage(json.dumps(msg))

        persons, confidences = self.infer(imgF, args)
        print('im don to check');
        print persons
        msg = {
                "type": "IDENTITIES",
                "identities": persons
        }
        self.sendMessage(json.dumps(msg))

    def getRep(self, bgrImg):
        print('im get rep')
        start = time.time()
        if bgrImg is None:
            raise Exception("Unable to load image/frame")
        imagem= cv2.imread('/Users/xinfangyuan/Ai/openface/demos/web/hg/myphoto.jpg')
        rgbImg = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

      
        print("  + Original size: {}".format(rgbImg.shape))
      

        start = time.time()

        # Get the largest face bounding box
        # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

        # Get all bounding boxes
        bb = align.getAllFaceBoundingBoxes(rgbImg)

        if bb is None:
            # raise Exception("Unable to find a face: {}".format(imgPath))
            return None
     
        print("Face detection took {} seconds.".format(time.time() - start))

        start = time.time()

        alignedFaces = []
        for box in bb:
            alignedFaces.append(
                align.align(
                    args.imgDim,
                    rgbImg,
                    box,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

        if alignedFaces is None:
            raise Exception("Unable to align the frame")
      
        print("Alignment took {} seconds.".format(time.time() - start))

        start = time.time()

        reps = []
        for alignedFace in alignedFaces:
            reps.append(net.forward(alignedFace))


        print("Neural network forward pass took {} seconds.".format(
                time.time() - start))

        # print (reps)
        return reps

    def infer(self, img, args):
        print('im here')
        file = "/Users/xinfangyuan/Ai/openface/generated-embeddings/classifier.pkl"
        with open(file, 'r') as f:
            if sys.version_info[0] < 3:
                    (le, clf) = pickle.load(f)  # le - label and clf - classifer
            else:
                    (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

        reps = self.getRep(img)
        persons = []
        confidences = []
        for rep in reps:
            try:
                rep = rep.reshape(1, -1)
            except:
                print ("No Face detected")
                return (None, None)
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            # print (predictions)
            maxI = np.argmax(predictions)
            # max2 = np.argsort(predictions)[-3:][::-1][1]
            persons.append(le.inverse_transform(maxI))
            # print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
            # ^ prints the second prediction
            confidences.append(predictions[maxI])
            # if args.verbose:
            #     print("Prediction took {} seconds.".format(time.time() - start))
            #     pass
            # # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
            # if isinstance(clf, GMM):
            #     dist = np.linalg.norm(rep - clf.means_[maxI])
            #     print("  + Distance from the mean: {}".format(dist))
            #     pass
        print persons
        print confidences
        return (persons, confidences)



def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
