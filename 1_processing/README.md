## OCT Volume Preprocessing

`oct_to_mat.m` is the **main function** used for preprocessing OCT volumes.

All other `.m` files (`OCTFileClose.m`, `OCTFileGetChirp.m`, `OCTFileGetColoredData.m`, `OCTFileGetIntensity.m`, `OCTFileGetNrRawData.m`, `OCTFileGetProperty.m`, `OCTFileGetRawData.m`, `OCTFileGetRealData.m`, `OCTFileGetVariance.m`, `OCTFileOpen.m` and `xml2struct.m`) are **auxiliary functions** required to run the preprocessing pipeline (Thorlabs Ganymede Series, Thorlabs GmbH, Luebeck, Germany).  

These auxiliary functions and the file to be converted **must be located in the same directory** as `oct_to_mat.m` in order to work correctly. 