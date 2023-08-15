# Jina Demo
## Simple Video Classifier as JINA executor  

This is an example of a simple JINA executor that classifies videos and images

The demo is for [an evening of python coding meetup event](https://github.com/Jacob-Barhak/EveningOfPythonCoding).

To run this, install all requirements in the `requirements.txt` file using conda or pip - use python 3.10

This was tested on Python 3.10.12 | packaged by conda-forge | (main, Aug 22 2022, 20:29:51) [MSC v.1929 64 bit (AMD64)] on win32

## Usage

0. Copy the file imagenet_class_index.json from https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json and place it in the same directory as main.py - this file is under MIT license which is different than the CC0 license under which this code is distributed in - so be careful not to redistribute without changing the license. 
1. Place some images and/or videos to be classified in the `images` subdirectory
2. Open a terminal and execute the Jina server using the command: `python main.py` 
3. Open another terminal and execute the client using the command: `python client.py`

The classifications of the images / videos should be printed on the screen


## License
<a rel="license" href="http://creativecommons.org/publicdomain/zero/1.0/"> <img src="https://licensebuttons.net/p/zero/1.0/88x31.png" style="border-style: none;" alt="CC0" />  </a>

To the extent possible under law, Jacob Barhak has waived all copyright and related or neighboring rights to Jina Demo This work is published from: Israel.

Exceptions are:
* The file imagenet_class_index.json is downloaded from https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json and was originally under MIT license
* Libraries used outside the code provided here have their own licenses.
