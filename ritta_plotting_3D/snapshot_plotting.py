# trace generated using paraview version 5.13.3
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
mergeVectorComponents1 = FindSource('MergeVectorComponents1')

# create a new 'Python Calculator'
pythonCalculator1 = PythonCalculator(registrationName='PythonCalculator1', Input=mergeVectorComponents1)

# find source
flowTime_2048hdf5 = FindSource('flowTime_2048.hdf5')

# find source
cellDatatoPointData1 = FindSource('CellDatatoPointData1')

# Properties modified on pythonCalculator1
pythonCalculator1.Expression = 'mag(Vorticity) / max(mag(Vorticity))'
pythonCalculator1.ArrayName = 'normalized_vorticity'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
pythonCalculator1Display = Show(pythonCalculator1, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
pythonCalculator1Display.Representation = 'Outline'

# hide data in view
Hide(mergeVectorComponents1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=pythonCalculator1)

# Properties modified on contour1
contour1.ContourBy = ['POINTS', 'normalized_vorticity']
contour1.Isosurfaces = [0.02]

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'normalized_vorticity'
normalized_vorticityLUT = GetColorTransferFunction('normalized_vorticity')

# get opacity transfer function/opacity map for 'normalized_vorticity'
normalized_vorticityPWF = GetOpacityTransferFunction('normalized_vorticity')

# get 2D transfer function for 'normalized_vorticity'
normalized_vorticityTF2D = GetTransferFunction2D('normalized_vorticity')

# hide data in view
Hide(pythonCalculator1, renderView1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1018, 1652)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [16.541000915603693, -10.843439593027883, -12.919194511662816]
renderView1.CameraFocalPoint = [3.062499999999999, 6.839581379855959e-16, 1.1908199723856357e-16]
renderView1.CameraViewUp = [-0.3652158467802428, 0.48947243848399624, -0.791854858686313]
renderView1.CameraParallelScale = 12.235410568129746


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------