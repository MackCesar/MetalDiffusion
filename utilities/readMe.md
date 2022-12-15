# Utilities

## modelFinder.py
**Model Finder** is a script that locates the models, .ckpt's, and .h5's for the program.

Currently, the script can only find .ckpt's.

## readWriteFile.py
**Read and Write to File** is a script that writes data into .txt files.

Currently, it is used to write:
1) the art/cinema creation settings of a generation. The input text is a list of strings that are written into a file. Every index is a line of data
2) the analysis of a .ckpt file. This was used in the development of SD2.x adoption

## settingsControl.py
**User Settings Control** is a script that reads and writes the user preferences and settings.

This script contains the factory defaults for the webUI.

## videoUtilities.py
**Video Utilities** is a script that handles all things regarding video creation.

Currently, the zoom out function isn't working as intended. Beware!
