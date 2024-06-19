import numpy as np

class Content:
	def __init__(self,location):
		self.path = location
		self.energy = None
		self.zenith = None
		self.azimuth = None
class Library:
	def __init__(self,table_of_content):
		self.content_list = table_of_content
	
