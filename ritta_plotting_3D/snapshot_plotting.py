# trace generated using paraview version 5.13.3
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
flowTime_2048hdf5 = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
flowTime_2048hdf5Display = Show(flowTime_2048hdf5, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
flowTime_2048hdf5Display.Representation = 'Outline'

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on flowTime_2048hdf5
flowTime_2048hdf5.CellArrayStatus = ['u_0', 'u_1', 'u_2']

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=flowTime_2048hdf5)

# show data in view
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
cellDatatoPointData1Display.Representation = 'Outline'

# hide data in view
Hide(flowTime_2048hdf5, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Merge Vector Components'
mergeVectorComponents1 = MergeVectorComponents(registrationName='MergeVectorComponents1', Input=cellDatatoPointData1)

# Properties modified on mergeVectorComponents1
mergeVectorComponents1.OutputVectorName = 'Velocity'

# show data in view
mergeVectorComponents1Display = Show(mergeVectorComponents1, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
mergeVectorComponents1Display.Representation = 'Outline'

# hide data in view
Hide(cellDatatoPointData1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Gradient'
gradient1 = Gradient(registrationName='Gradient1', Input=mergeVectorComponents1)

# Properties modified on gradient1
gradient1.ComputeQCriterion = 1

# show data in view
gradient1Display = Show(gradient1, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
gradient1Display.Representation = 'Outline'

# hide data in view
Hide(mergeVectorComponents1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=gradient1)

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'QCriterion'
qCriterionLUT = GetColorTransferFunction('QCriterion')

# get opacity transfer function/opacity map for 'QCriterion'
qCriterionPWF = GetOpacityTransferFunction('QCriterion')

# get 2D transfer function for 'QCriterion'
qCriterionTF2D = GetTransferFunction2D('QCriterion')

# set active source
SetActiveSource(gradient1)

# set active source
SetActiveSource(contour1)

# hide data in view
Hide(gradient1, renderView1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(802, 1033)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [16.65225298909394, 0.7536788379147482, -4.296194431796246]
renderView1.CameraFocalPoint = [3.0625000000000027, -1.0977953873560387e-15, 1.807254991429669e-15]
renderView1.CameraViewUp = [-0.016372486611386468, 0.9923582196167854, 0.12229924628207733]
renderView1.CameraParallelScale = 9.711408782056534


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