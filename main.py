#!/usr/bin/env python
#
# Copyright 2007 Google Inc.
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
#
import webapp2
from django.utils import simplejson as json
import numpy as np
import logging

class MainHandler(webapp2.RequestHandler):
    def get(self):
        self.response.write('Please post to this endpoint!')
    
    def post(self):
        jsonstring = self.request.body
        jsonobject = json.loads(jsonstring)
        self.response.headers['Content-Type'] = 'application/json'
        result = GetPos(jsonobject['rssi'], jsonobject['pos'], jsonobject['accuracy'], jsonobject['method'])
        obj = {
            'result': result.tolist()
        }
        self.response.out.write(json.dumps(obj))

def GetPos(rssi, pos, accuracy, method):
    """Calculate position of mobile device.
        
        rssi: rssi in dB, N * 1 array
        
        pos: position of N APs, N * 3 array
        
        accuracy: accuray of AP position in meters, N * 1 array
        
        method: a 2-char string like 'A1', 'C4', ...
        the first char:
        'A': non-iterative algorithms
        'B': iterative algorithm
        'C': iterative algorithm with outlier removal
        the second char is explained in the code
        """
    # path loss model:
    #   rssi = rssi0 - 10 * pathLossExp * log10(distance)
    rssi0 = -34
    pathLossExp = 3.3
    
    accLmt = 0.01   # threshold to terminate iteration
    
    if method.startswith('A'):
        if method.endswith('1'):    # Simple median
        #return stats.nanmedian(pos)
            return np.median(pos,axis=0)
        elif method.endswith('2'):  # Simple mean
            return np.mean(pos,axis=0)
        elif method.endswith('3'):  # Weighted mean
            weight = np.exp(np.array(rssi) / 10.0) / np.array(accuracy)
            return np.average(pos, 0, weight)
    elif method.startswith('B'):
        nMeasurements = len(rssi)
        weight = np.ones(nMeasurements)
        estPos3d = np.average(pos, 0, weight)
        if nMeasurements < 4:
            return estPos3d
        pos2d = pos[:, 0:2]
        estPos2d = estPos3d[0:2]
        r = 10.0 ** ((rssi0 - rssi) / (10.0 * pathLossExp))
        for i in xrange(10):
            estRangeVec = pos2d - estPos2d
            estRange = np.hypot(estRangeVec[:, 0], estRangeVec[:, 1])
            dRange = estRange - r
            # geometric matrix
            G = (estRangeVec.T / estRange).T
            if method.endswith('1'):    # Ordinary LS
                dEstPos = np.linalg.lstsq(G, dRange)[0]
            elif method.endswith('5'):  # L1 minimization via linear programming
                raise Exception('Not implemented yet!')
            estPos2d += dEstPos
            if np.fabs(dEstPos).max() < accLmt:
                break
        else:
            logging.error('Iteration did not converge.')
        estPos3d[0:2] = estPos2d
        return estPos3d
    logging.error('Unsupported method: {}.'.format(method))
    return None

app = webapp2.WSGIApplication([
    ('/', MainHandler)
], debug=True)
