"""Script for checking the correct versions of Python, StrawberryFields and TensorFlow are being
used."""
import sys

import strawberryfields as sf
import tensorflow as tf

python_version = sys.version_info
sf_version = sf.__version__
tf_version = tf.__version__.split(".")

if python_version > (3, 6):
    raise SystemError("Your version of python is {}.{}. You must have Python 3.5 or 3.6 installed "
                      "to run this script.".format(python_version.major, python_version.minor))

if python_version < (3, 5):
    raise SystemError("Your version of python is {}.{}. You must have Python 3.5 or 3.6 installed "
                      "to run this script.".format(python_version.major, python_version.minor))

if sf_version != "0.10.0":
    raise ImportError("An incompatible version of StrawberryFields is installed. You must have "
                      "StrawberryFields version 0.10 to run this script. To install the correct "
                      "version, run:\n >>> pip install strawberryfields==0.10")

if not(tf_version[0] == "1" and tf_version[1] == "3"):
    raise ImportError("An incompatible version of TensorFlow is installed. You must have "
                      "TensorFlow version 1.3 to run this script. To install the correct "
                      "version, run:\n >>> pip install tensorflow==1.3")
