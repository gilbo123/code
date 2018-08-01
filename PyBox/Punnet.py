#!/usr/bin/python

import numpy as np
import PyCapture2
from timeit import default_timer as timer


class Punnet:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some members
        self.RGBTopImage = None
        self.IRTopImage = None
        self.RGBBtmImage = None
        self.IRBtmImage = None

        self.status = 'Pass'
        self.dateTime = None
        self.punnetNeedsProcessing = False
        self.punnetNeedsDisplaying = False

    def resetPunnet():
        self.RGBTopImage = None
        self.IRTopImage = None
        self.RGBBtmImage = None
        self.IRTopImage = None

        self.status = 'Pass'
        self.dateTime = None
        self.punnetNeedsProcessing = False
        self.punnetNeedsDisplaying = False
