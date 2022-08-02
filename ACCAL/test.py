#!/usr/bin/python

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

import Image.processing

Image.processing.cropToCoin()
