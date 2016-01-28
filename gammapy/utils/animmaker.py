# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility class to export animations from Gammapy for visualization"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import numpy as np

class AnimMaker:
	"""
	Utility class to export animation from NumPy arrays
	
	The purpose of this class is to generate an animation from a specified
	sequence of NumPy arrays. Additionally, parameters like frame rate and
	encoder choice can be specified.
	
	Attributes
	----------

	imlist : tuple of ndarrays
		The input list of ndarrays is stored in this attribute
	
	fig : Pyplot figure
		Container for the figure that is being generated

	im : Pyplot image
		Container for the pyplot image:w

	curr_frame : int
		Frame number of the currently displayed frame

	ani : matplotlib.animation.FuncAnimation object 
		Animation object for the current animation

	show_anim : function
		Show the generated animation with the given parameters

	write_anim : function
		Write the animation to file with the specified class
		parameters

	Parameters
	----------

	imlist : tuple of ndarrays 
		Input image NumPy ndarrays in the order in which they
		have to be put in the output animation.
	"""

	def __init__(self, imlist, duration=200):
		self.fig = plt.figure()
		self.imlist = imlist
		self.im = plt.imshow(self.imlist[0], animated=True)
		plt.axis('off')
		self.curr_frame = 0

		def updatefig(*args):
			self.curr_frame = (self.curr_frame + 1)%len(self.imlist)
			self.im.set_array(self.imlist[self.curr_frame])

		self.ani = animation.FuncAnimation(self.fig, updatefig, interval=duration, blit=True)

	def show_anim(self):
		"""
		Show the generated animation
		"""
		plt.show()

	def write_anim(self, fname, enc='mencoder'):
		"""
		Write the image list to file 

		Parameters
		----------
		
		fname : string
			Output file name
		duration : float, optional
			Time (ms) between consecutive animation frames
		"""
		self.ani.save(fname, writer=enc)
