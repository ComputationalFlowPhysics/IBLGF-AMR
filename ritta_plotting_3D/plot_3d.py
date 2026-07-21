#!/usr/bin/env pvpython

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

from paraview.simple import *
import paraview

# args : output folder path, stride
# save outputs in outputs folder

if __name__ == "__main__":
    for each output timeframe: 
        paraview.simple._DisableFirstRenderCameraReset()
        # filename : output folder path / output / ___.hdf5
        source = VisItChomboReader(registrationName='', FileName=[''])
        renderView1.ResetCamera(False, 0.9)
        materialLibrary1 = GetMaterialLibrary()

        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.Update()

        source.CellArrayStatus = ['u_0', 'u_1', 'u_2']
        renderView1.Update()

        cellDatatoPointData = CellDatatoPointData(registrationName='CellDatatoPointData', Input=source)
        renderView1.Update()

        mergeVectorComponents = MergeVectorComponents(registrationName='MergeVectorComponents', Input=cellDatatoPointData)
        mergeVectorComponents.XArray = 'u_0'
        mergeVectorComponents.YArray = 'u_1'
        mergeVectorComponents.ZArray = 'u_2'
        mergeVectorComponents.OutputVectorName = 'Velocity'
        renderView1.Update()

        gradient = Gradient(registrationName='Gradient', Input=mergeVectorComponents)
        renderView1.Update()

        contour1 = Contour(registrationName='Contour1', Input=gradient1)
