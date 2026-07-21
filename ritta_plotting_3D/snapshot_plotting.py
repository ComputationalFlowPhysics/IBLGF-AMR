# trace generated using paraview version 5.13.3
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'VisIt Chombo Reader'
flowTime_2048hdf5 = VisItChomboReader(registrationName='flowTime_2048.hdf5', FileName=['/Users/rittachoi/Desktop/Caltech/CFD Lab/IBLGF/IBLGF-AMR/runs/ns_amr_lgf/flowTime_2048.hdf5'])

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
mergeVectorComponents1.YArray = 'u_1'
mergeVectorComponents1.ZArray = 'u_2'
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

# Properties modified on contour1
contour1.Isosurfaces = [2.5]

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
layout1.SetSize(903, 899)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [17.139627675282647, 11.400499110766626, -22.814391556735938]
renderView1.CameraFocalPoint = [3.0625, 0.0, 0.0]
renderView1.CameraViewUp = [-0.12585964286795603, 0.9162931489586358, 0.38021864166373776]
renderView1.CameraParallelScale = 7.539738473581163


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
