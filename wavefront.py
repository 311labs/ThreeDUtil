#!/usr/bin/python

import os, errno
import os.path
import sys
import shutil
import numpy as np
import math

import Image, ImageDraw
from ImageColor import getcolor, getrgb
from ImageOps import grayscale

OBJ_HEADER = "# ThreeD Me - OBJ File\n"
MTL_HEADER = "# ThreeD Me - MTL File\n"

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

def distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

class Mesh(object):
	"""Our generic mesh"""
	def __init__(self, name):
		self.name = name
		self.ordered_materials = []
		self.materials = {}
		self.v_start = -1
		self.v_end = 0
		self.vt_start = -1
		self.vt_end = 0
		self.vertices = []
		self.normals = None
		self.tex_coords = []

	def faceCount(self):
		count = 0
		for m in self.ordered_materials:
			count += len(m.faces)
		return count

	def materialCount(self):
		return len(self.ordered_materials)		

	def textureCount(self):
		count = 0
		for m in self.ordered_materials:
			count += len(m.textures.keys())
		return count

	def addMaterial(self, material):
		self.ordered_materials.append(material)
		self.materials[material.name] = material

	def calculateDimensions(self):
		count = 0
		vt_count = 0
		vn_count = 0
		self.width = 0
		self.height = 0
		self.depth = 0
		self.min_x = None
		self.max_x = None
		self.min_y = None
		self.max_y = None
		self.min_z = None
		self.max_z = None

		self.total_vertices = len(self.vertices)
		self.total_texture_coordinates = len(self.tex_coords)
		#self.total_normals = len(self.normals)
		for v in self.vertices:
			self.minmax_x(v[0])
			self.minmax_y(v[1])
			self.minmax_z(v[2])

		self.width = self.max_x - self.min_x
		self.height = self.max_y - self.min_y
		self.depth = self.max_z - self.min_z

	def minmax_x(self, x):
		if self.max_x is None:
			self.max_x = x
			self.min_x = x
		if x > self.max_x:
			self.max_x = x
		if x < self.min_x:
			self.min_x = x

	def minmax_y(self, y):
		if self.max_y is None:
			self.max_y = y
			self.min_y = y
		if y > self.max_y:
			self.max_y = y
		if y < self.min_y:
			self.min_y = y

	def minmax_z(self, z):
		if self.max_z is None:
			self.max_z = z
			self.min_z = z
		if z > self.max_z:
			self.max_z = z
		if z < self.min_z:
			self.min_z = z

class Material(object):
	def __init__(self, name=''):
		self.ordered_properties = ["Ns", "Ka", "Kd", "Ks", "Ni", "d", "illum", "map_Ka", "map_Kd", "map_Ns", "map_d", "map_bump", "bump", "disp", "decal"]
		for prop in self.ordered_properties:
			setattr(self, prop, None)
		self.name = name
		self.faces = []
		self.facemap = {}
		self.diffuse = [0.8, 0.8, 0.8, 1.0]
		self.ambient = [0.2, 0.2, 0.2, 1.0]
		self.specular = [0.0, 0.0, 0.0, 1.0]
		self.emissive = [0.0, 0.0, 0.0, 1.0]
		self.shininess = 0.0
		self.vertices = []
		self.textures = {}

	def set_alpha(self, alpha):
		"""Set alpha/last value on all four lighting attributes."""
		alpha = [float(alpha)]
		self.diffuse = self.diffuse[:3] + alpha 
		self.ambient = self.ambient[:3] + alpha
		self.specular = self.specular[:3] + alpha
		self.emissive = self.emissive[:3] + alpha

	def set_ambient(self, values):
		self.ambient = values

	def set_specular(self, values):
		self.specular = values
	
	def set_emmissive(self, values):
		self.emmissive = values
	
	def set_diffuse(self, values):
		self.diffuse = values

	def addTexture(self, texture):
		self.textures[texture.name] = texture

class Texture(object):
	"""Texture File"""
	def __init__(self, filename, root=None):
		self.name = filename
		self.subpath = os.path.dirname(filename)
		self.basename = os.path.basename(filename)
		self.ext = self.basename.split('.')[1]
		self.filename = filename
		self.root = root
		self.img = None
		self.fullpath = os.path.join(os.path.abspath(root), filename)

	def open(self):
		self.img = Image.open(self.fullpath)

	def save(self):
		self.img.save(self.fullpath, overwrite=True)

	def close(self):
		self.img = None

	def getSize(self):
		return self.getDimensions()

	def getDimensions(self):
		if self.img:
			return self.img.size

		img = Image.open(self.fullpath)
		self.size = img.size
		return self.size

	def resize(self, size):
		self.open()
		self.img.resize(size)
		self.save()
		self.close()

	def circle(self, tl, br, fill, gradient=True):
		# create a vertical gradient...
		width = br[0] - tl[0]
		height = br[1] - tl[1]

		dim = 600
		dim1 = dim-1
		d = dim/2
		gradient = Image.new('L', (dim,dim))
		for x in range(dim):		
			for y in range(dim):
				c = math.hypot(x - d, y - d)
				if c < 150:
					c = 0
				else:
					c -= 150
					c = c * 3
				c = 255-c
				if c < 0:
					c = 0
				gradient.putpixel((x,y),c)
		# first create a rectangle image with our skin color
		rect = Image.new("RGBA", (width, height), fill)
		# resize the gradient to the size of im...
		alpha = gradient.resize(rect.size)
		# put alpha in the alpha band of im...
		rect.putalpha(alpha)
		self.open()
		# check if im has Alpha band...
		if self.img.mode != 'RGBA':
			self.img = self.img.convert('RGBA')

		self.img.paste(rect, (tl[0],tl[1],br[0],br[1]), rect)
		self.img = self.img.convert('RGBA')
		self.save()
		self.close()

	def paste(self, img):
		if type(img) in [str, unicode]:
			img = Image.open(img)

		self.open()
		width,height = img.size
		self.img.paste(img, (0,0,width,height), img)
		self.save()
		self.close()

	def rectangle(self, tl, br, fill, gradient="tb"):
		# create a vertical gradient...
		width = br[0] - tl[0]
		height = br[1] - tl[1]

		gradient = Image.new('L', (1,255))
		for y in range(255):
			gradient.putpixel((0,254-y),y)

		if gradient == "bt":
			gradient.rotate(180)
		elif gradient == "lr":
			gradient.rotate(90)
		elif gradient == "rl":
			gradient.rotate(270)
		# first create a rectangle image with our skin color
		rect = Image.new("RGB", (width, height), fill)
		# resize the gradient to the size of im...
		alpha = gradient.resize(rect.size)
		# put alpha in the alpha band of im...
		rect.putalpha(alpha)
		self.open()
		# check if im has Alpha band...
		if self.img.mode != 'RGBA':
			self.img = self.img.convert('RGBA')

		self.img.paste(rect, (tl[0],tl[1],br[0],br[1]), rect)
		self.img = self.img.convert('RGB')
		self.save()
		self.close()

	def tint(self, tint='#ffffff'):
		self.open()
		src = self.img
		if src.mode not in ['RGB', 'RGBA']:
			raise TypeError('Unsupported source image mode: {}'.format(src.mode))
		src.load()

		if type(tint) in [str, unicode]:
			tr, tg, tb = getrgb(tint)
			tl = getcolor(tint, "L")  # tint color's overall luminosity
		else:
			tr, tg, tb = tint
			tl = sum([tr,tg,tb])/3

		if not tl: tl = 1  # avoid division by zero
		tl = float(tl)  # compute luminosity preserving tint factors
		sr, sg, sb = map(lambda tv: tv/tl, (tr, tg, tb))  # per component adjustments

		# create look-up tables to map luminosity to adjusted tint
		# (using floating-point math only to compute table)
		luts = (map(lambda lr: int(lr*sr + 0.5), range(256)) +
				map(lambda lg: int(lg*sg + 0.5), range(256)) +
				map(lambda lb: int(lb*sb + 0.5), range(256)))
		l = grayscale(src)  # 8-bit luminosity version of whole image
		if Image.getmodebands(src.mode) < 4:
			merge_args = (src.mode, (l, l, l))  # for RGB verion of grayscale
		else:  # include copy of src image's alpha layer
			a = Image.new("L", src.size)
			a.putdata(src.getdata(3))
			merge_args = (src.mode, (l, l, l, a))  # for RGBA verion of grayscale
			luts += range(256)  # for 1:1 mapping of copied alpha values

		self.img = Image.merge(*merge_args).point(luts)
		self.save()
		self.close()

	def normalizeColor(self, newcolor, range=40):
		# normalize the texture color to this new color
		im = Image.open('test.png')
		im = im.convert('RGBA')

		data = np.array(im)   # "data" is a height x width x 4 numpy array
		red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

		# Replace white with red... (leaves alpha values alone...)
		white_areas = (red == 255) & (blue == 255) & (green == 255)
		data[..., :-1][white_areas] = (255, 0, 0)

		im2 = Image.fromarray(data)
		im2.show()

	def exists(self):
		return os.path.exists(self.fullpath)

	def moveTo(self, root, filename=None):
		if filename is None:
			filename = self.filename
		newroot = root
		subpath = os.path.dirname(filename)
		basename = os.path.basename(filename)
		fullpath = os.path.join(os.path.abspath(root), filename)
		if os.path.exists(fullpath):
			print "texture already exists '{0}'".format(fullpath)
		else:
			if subpath:
				try:
					os.makedirs(os.path.join(newroot, subpath))
				except:
					pass
			print "copying textures: {0} -> {1}".format(self.fullpath, fullpath)
			shutil.copy2(self.fullpath, fullpath)
		self.name = filename
		self.subpath = ""
		self.basename = os.path.basename(basename)
		self.ext = self.basename.split('.')[1]
		self.filename = filename
		self.root = root
		self.img = None
		self.fullpath = os.path.join(os.path.abspath(root), filename)
		
class File(object):
	def __init__(self, filename):
		self.fullpath = os.path.abspath(filename)
		self.filename = filename
		self.root = os.path.dirname(self.fullpath)
		self.basename = os.path.basename(filename)

	def info(self):
		# print info about the file
		print self.filename

	def on_load_complete(self):
		pass

	def load(self):
		"""
		Loads the file into memory
		"""
		f = open(self.fullpath, 'r')
		while True:
			line = f.readline().decode('ascii', 'ignore')
			if not line:
				break
			if line.startswith("#"):
				continue
			values = line.split()
			if len(values) > 1:
				getattr(self, "parse_{0}".format(values[0]))(values[1:])
		f.close()
		self.on_load_complete()

	def save(self, filename=None, merge=[]):
		pass	

class MTLFile(File):
	def __init__(self, filename, root=None):
		self.name = filename
		if root:
			filename = os.path.join(root, filename)
		super(MTLFile, self).__init__(filename)
		self.root = root
		self.textures = {}
		self.materials = {}
		self.cur_material = None

	def addMaterial(self, mat):
		if type(mat) in [str, unicode]:
			self.cur_material = Material(mat)
			self.materials[mat] = self.cur_material		
		else:
			for key in mat.textures:
				if not self.textures.has_key(key):
					self.textures[key] = mat.textures[key]
			self.materials[mat.name] = mat
			self.cur_material = mat

	def getTexture(self, filename):
		if self.textures.has_key(mat):
			return self.textures[mat]
		return None		

	def getMaterial(self, mat):
		if self.materials.has_key(mat):
			return self.materials[mat]
		return None

	def parse_newmtl(self, args):
		self.addMaterial(args[0])

	def parse_Kd(self, args):
		self.cur_material.Kd = map(float, args)
		self.cur_material.set_diffuse(args)

	def parse_Ka(self, args):
		self.cur_material.Ka = map(float, args)
		self.cur_material.set_ambient(args)

	def parse_Ks(self, args):
		self.cur_material.Ks = map(float, args)
		self.cur_material.set_specular(args)

	def parse_Ke(self, args):
		self.cur_material.Ke = args
		self.cur_material.set_emissive(args)

	def parse_Ns(self, args):
		self.cur_material.Ns = float(args[0])
		self.cur_material.shininess = float(args[0])

	def parse_d(self, args):
		self.cur_material.d = float(args[0])
		self.cur_material.set_alpha(args[0])

	def parse_Tr(self, args):
		self.parse_d(args)

	def parse_Ni(self, args):
		self.cur_material.Ni = float(args[0])

	def parse_illum(self, args):
		self.cur_material.illum = int(args[0])

	def parse_map_Ka(self, args):
		self.cur_material.map_Ka = self.loadTexture(args[0])

	def parse_map_Ka(self, args):
		self.cur_material.map_Ka = self.loadTexture(args[0])

	def parse_map_Kd(self, args):
		self.cur_material.map_Kd = self.loadTexture(args[0])

	def parse_map_Ks(self, args):
		self.cur_material.map_Ks = self.loadTexture(args[0])

	def parse_map_Ns(self, args):
		self.cur_material.map_Ns = self.loadTexture(args[0])

	def parse_map_D(self, args):
		self.cur_material.map_d = self.loadTexture(args[0])

	def parse_map_d(self, args):
		self.cur_material.map_d = self.loadTexture(args[0])

	def parse_map_bump(self, args):
		self.cur_material.map_bump = self.loadTexture(args[0])

	def loadTexture(self, filename):
		if self.textures.has_key(filename):
			return self.textures[filename]
		tex_file = Texture(filename, self.root)
		self.textures[filename] = tex_file
		self.cur_material.addTexture(tex_file)
		return tex_file

	def saveMaterial(self, mat):
		self.f.write("newmtl {0}\n".format(mat.name))
		for prop in mat.ordered_properties:
			value = getattr(mat, prop)
			if value != None:
				if type(value) in [str, int, float]:
					self.f.write("{0} {1}\n".format(prop, value))
				elif type(value) in [list, tuple]:
					self.f.write("{0} {1}\n".format(prop, " ".join(map(str, value))))
				elif type(value) is Texture:
					self.f.write("{0} {1}\n".format(prop, value.name))
		self.f.write("\n")

	def merge(self, merge):
		# handle merging
		for m in merge:
			print "merging {0} into {1}".format(m.name, self.name)
			for mat in m.materials.values():
				if self.materials.has_key(mat.name):
					print "Merge warning! Overwriting Material {0}".format(mat.name)
				self.materials[mat.name] = mat

			for tex in m.textures.values():
				if self.textures.has_key(tex.name):
					print "Merge warning! Overwriting Texutre {0}".format(tex.name)
				self.textures[tex.name] = tex

	def save(self, filename=None, merge=[]):
		self.name = os.path.basename(filename)
		self.fullpath = os.path.abspath(filename)
		root = os.path.dirname(self.fullpath)
		flatten = False
		# print "{0} vs {1}".format(root, self.root)
		if root != self.root:
			flatten = True
		self.root = root

		self.filename = self.fullpath

		self.merge(merge)
		for texture in self.textures.values():
			if not texture.exists():
				print("TEXTURE ERROR: Original file does not exists '{0}'".format(texture.fullpath))
				continue
			if flatten:
				texture.moveTo(self.root, os.path.basename(texture.filename))
			else:
				texture.moveTo(self.root, texture.filename)	

		print "saving MTL File: '{0}'".format(self.filename)
		print "\twith {0} materials {1} textures".format(len(self.materials.values()), len(self.textures.values()))
		self.f = open(self.fullpath, 'w')
		self.f.write(MTL_HEADER)
		self.f.write("# Material Count: {0}\n\n".format(len(self.materials.keys())))
		for mat in self.materials.values():
			self.saveMaterial(mat)
		self.f.close()

class OBJFile(File):
	def __init__(self, filename):
		super(OBJFile, self).__init__(filename)
		self.meshes = {}
		self.ordered_meshes = []
		self.mesh = None
		self.material = None
		self.vertices = []
		self.normals = []
		self.tex_coords = []

		self.face_vertices = []
		self.face_normals = []
		self.face_coords = []
		self.corrected_faces = []
		self.facemap = {}

		self.mtllib = None

	def getTextureFiles(self):
		return self.mtllib.textures.values()

	def getMesh(self, name):
		return self.meshes[name]

	def addMesh(self, mesh):
		add_material = self.mesh is None and self.material
		if type(mesh) in [str, unicode]:
			self.mesh = Mesh(mesh)
		else:
			self.mesh = mesh
			for mat in self.mesh.ordered_materials:
				if self.mtllib is None:
					self.mtllib = MTLFile("temp.mtl")
				self.mtllib.addMaterial(mat)

		self.meshes[self.mesh.name] = self.mesh
		self.ordered_meshes.append(self.mesh)
		if add_material:
			self.mesh.addMaterial(self.material)

	def addMaterial(self, mat):
		if type(mat) in [str, unicode]:
			self.material = self.mtllib.getMaterial(mat)
			if self.material is None:
				raise Exception("material was not found {0}".format(mat))
		else:
			self.material = mat
		if self.mesh:
			self.mesh.addMaterial(self.material)

	def parse_v(self, args):
		self.vertices.append(map(float, args))

	def parse_vn(self, args):
		self.normals.append(map(float, args))

	def parse_vt(self, args):
		self.tex_coords.append(map(float, args))

	def parse_mtllib(self, args):
		self.mtllib = MTLFile(args[0], self.root)
		self.mtllib.load()

	def parse_usemtl(self, args):
		# set the mesh to none
		self.mesh = None
		self.addMaterial(args[0])

	def parse_o(self, args):
		self.parse_g(args)

	def parse_g(self, args):
		self.addMesh(args[0])

	def _absint(self, i, ref):
		i = int(i)
		if i>0 :
			return i-1
		else:
			print "read 0 value"
			return ref+i

	def parse_f(self, args):
		"""		
		a face is made up of index sets matching the position
		in the already parsed vertices, normals, and texture coordinates
		the format is one of the following:
			just the vertices
			f v1 v2 v3 ...
			vertices and texture coordinates
			f v1/vt1 v2/vt2 v3/vt3
			vertices, texture coordinates, and normals
			f v1/vt1/vn1 v2/vt2/vn3 v3/vt3/vn3
			vertices and normlas
			f v1//vn1 v2/vt2/vn3 v3//vn3
			
		"""
		if self.mesh is None:
			if len(self.ordered_meshes):
				# if we have meshes this means this is another usemtl in the same group
				self.mesh = self.ordered_meshes[-1]
			else:
				self.addMesh(self.material.name)
			self.mesh.addMaterial(self.material)
		if self.material is None:
			self.addMaterial("")
		face = [map(int, j) for j in [i.split('/') for i in args]]
		self.material.faces.append(face)

		# shift indexes from starting at 1 to starting at 0
		final_face = []
		for index_set in args:			
			indices = [i for i in index_set.split('/')]
			# correct the index to match 0 as start not 1
			vertex_index = self._absint(indices[0], len(self.vertices))
			if self.mesh.v_start == -1 or vertex_index < self.mesh.v_start:
				self.mesh.v_start = vertex_index
			elif vertex_index > self.mesh.v_end:
				self.mesh.v_end = vertex_index

			self.mesh.vertices.append( int(indices[0]) )

			if self.mesh.tex_coords != None:
				if len(indices) > 1 and indices[1]:
					texcord_index = self._absint(indices[1], len(self.mesh.tex_coords))
					if self.mesh.vt_start == -1 or texcord_index < self.mesh.vt_start:
						self.mesh.vt_start = texcord_index
					elif texcord_index > self.mesh.vt_end:
						self.mesh.vt_end = texcord_index
					self.mesh.tex_coords.append(int(indices[1]))
				else:
					print "ignoring Face Texture Coordinates, they are not specified for all"
					self.mesh.tex_coords = None

			# TODO add normals

	def parse_s(self, args):
		self.material.smoothing = args[0]

	def on_load_complete(self):
		# called when the file is done loading so we can do some final processing
		nv = 0
		nvt = 0
		for mesh in self.ordered_meshes:
			maxv = -1
			for vt in mesh.tex_coords:
				if vt > maxv:
					maxv = vt
			#print "Max Value {0}".format(maxv)
			# lets try and map vertices to info each mesh based on materials
			mesh.vertices = self.vertices[mesh.v_start:mesh.v_end+1]
			if mesh.tex_coords != None:
				mesh.tex_coords = self.tex_coords[mesh.vt_start:mesh.vt_end+1]
			if mesh.normals != None:
				mesh.normals = self.normals[mesh.vn_start:mesh.vn_end]
			if nv > 0:
				# we need to normalize the faces to match the new vertice indexes by mesh
				for mtl in mesh.ordered_materials:
					for face in mtl.faces:
						for indices in face:
							indices[0] -= nv
							indices[1] -= nvt
			nv += len(mesh.vertices)
			nvt += len(mesh.tex_coords)

	def calculateDimensions(self):
		count = 0
		vt_count = 0
		vn_count = 0
		self.width = 0
		self.height = 0
		self.depth = 0
		self.min_x = None
		self.max_x = None
		self.min_y = None
		self.max_y = None
		self.min_z = None
		self.max_z = None

		self.total_vertices = len(self.vertices)
		self.total_texture_coordinates = len(self.tex_coords)
		self.total_normals = len(self.normals)
		for v in self.vertices:
			self.minmax_x(v[0])
			self.minmax_y(v[1])
			self.minmax_z(v[2])

		self.width = self.max_x - self.min_x
		self.height = self.max_y - self.min_y
		self.depth = self.max_z - self.min_z

	def minmax_x(self, x):
		if self.max_x is None:
			self.max_x = x
			self.min_x = x
		if x > self.max_x:
			self.max_x = x
		if x < self.min_x:
			self.min_x = x

	def minmax_y(self, y):
		if self.max_y is None:
			self.max_y = y
			self.min_y = y
		if y > self.max_y:
			self.max_y = y
		if y < self.min_y:
			self.min_y = y

	def minmax_z(self, z):
		if self.max_z is None:
			self.max_z = z
			self.min_z = z
		if z > self.max_z:
			self.max_z = z
		if z < self.min_z:
			self.min_z = z

	def filter_box(self, ltf, rbb, scale=None, trans=None):
		"""
		Clip all the vertices + faces found in the box
			ltf = Left Top Front (xyz)
			rbb = Right Bottom Back (xyz)
		"""
		print "filtering {0}, {1}".format(ltf, rbb)
		if type(scale) in [int, float]:
			scale = [scale, scale, scale]
		if type(trans) in [int, float]:
			trans = [trans, trans, trans]
		print scale
		vcount = 0
		for mesh in self.ordered_meshes:
			for v in mesh.vertices:
				if (ltf[0] <= v[0] <= rbb[0]) and (ltf[1] >= v[1] >= rbb[1]) and (ltf[2] >= v[2] >= rbb[2]):
					vcount += 1
					if scale:
						v[0] = v[0] * scale[0]
						v[1] = v[1] * scale[1]
						v[2] = v[2] * scale[2]
					if trans:
						v[0] = v[0] + trans[0]
						v[1] = v[1] + trans[1]
						v[2] = v[2] + trans[2]
		print "\t{0} vertices filtered".format(vcount)

	def getBottomBox(self, pct):
		self.calculateDimensions()
		max_y = self.min_y + (self.height * pct)
		ltf = (self.min_x-1,max_y,self.max_z+1)
		rbb = (self.max_x+1,self.min_y-1,self.min_z-1)
		return ltf, rbb		

	def getTopBox(self, pct):
		self.calculateDimensions()
		min_y = self.max_y - (self.height * pct)
		ltf = (self.min_x-1,self.max_y+1,self.max_z+1)
		rbb = (self.max_x+1,min_y,self.min_z-1)
		return ltf, rbb	

	def findFacesWithIndex(self, index, mesh):
		faces = []
		for mat in mesh.ordered_materials:
			for face in mat.faces:
				for indices in face:
					if indices[0] == index:	
						faces.append(face)
						break
		return faces

	def findLowestFaces(self, mesh, seen=[]):
		faces = []
		for mat in mesh.ordered_materials:
			for face in mat.faces:
				face = face

	def findLowestNeighbors(self, index, mesh, seen, high=False):
		faces = self.findFacesWithIndex(index, mesh)
		kept = []
		v = mesh.vertices[index-1]
		lv = None
		li = index
		lf = None
		# find lowest vertex
		for face in faces:
			if face not in seen:
				for indices in face:
					nv = mesh.vertices[indices[0]-1]
					if nv[1] != v[1]:
						if lv is None:
							lv = nv
							li = indices[0]
							lf = face
						elif high and nv[1] >= lv[1]:
							li = indices[0]
							lv = nv
							lf = face							
						elif not high and nv[1] <= lv[1]:
							li = indices[0]
							lv = nv
							lf = face
							#kept.append(face)
							#seen.append(face)
		if li == index:
			return None, None
		return li, face

	def join(self, mesh1, mesh2):
		mesh1.calculateDimensions()
		mesh2.calculateDimensions()
		# WE ONLY LOOK AT Y FOR NOW
		# TODO LOOK AT ALL AXIS
		top = mesh1
		bottom = mesh2
		if mesh1.max_y < mesh2.max_y:
			# this is lame, need to fix
			# hack to join bottom of mesh1 to top of mesh2
			top = mesh2
			bottom = mesh1

		# now find all of the faces at the bottom
		# work are way across
		min_y = top.min_y
		vindex = 1
		for v in top.vertices:
			if v[1] <= min_y:  
				break
			vindex += 1

		top_faces = []
		top_lowest = [vindex]
		face_count = top.faceCount()
		lcount = 0
		while True:
			vindex, face = self.findLowestNeighbors(vindex, top, top_faces)
			if vindex is None:
				break

			if lcount == len(top_faces):
				print "no change"
			lcount = len(top_faces)

			if len(top_faces) >= face_count:
				break
			top_lowest.append(vindex)
			top_faces.append(face)

		print "found {0} faces on top".format(len(top_faces))

		min_y = bottom.max_y
		vindex = 1
		for v in bottom.vertices:
			if v[1] >= min_y:  
				break
			vindex += 1

		print "bottom index"
		print vindex
		bottom_faces = []
		bottom_high = [vindex]
		while True:
			vindex, face = self.findLowestNeighbors(vindex, bottom, bottom_faces, True)
			if vindex is None:
				break
			bottom_high.append(vindex)
			bottom_faces.append(face)

		print "found {0} faces on bottom".format(len(bottom_faces))
		# now keep finding faces and choose miny face in each group



	def clipBottom(self, pct):
		ltf, rbb = self.getBottomBox(pct)
		self.clip(ltf, rbb)

	def clipTop(self, pct):
		ltf, rbb = self.getTopBox(pct)
		self.clip(ltf, rbb)

	def clip(self, ltf, rbb):
		"""
		Clip all the vertices + faces found in the box
			ltf = Left Top Front (xyz)
			rbb = Right Bottom Back (xyz)
		"""
		print "clipping {0}, {1}".format(ltf, rbb)
		clipped_v_indexes = []
		for mesh in self.ordered_meshes:
			index = 1
			index_map = {}
			new_verts = []
			for v in mesh.vertices:
				if (ltf[0] <= v[0] <= rbb[0]) and (ltf[1] >= v[1] >= rbb[1]) and (ltf[2] >= v[2] >= rbb[2]):
					# we are in the clip box
					clipped_v_indexes.append(index)
				else:
					new_verts.append(v)
					index_map[index] = len(new_verts)
				index += 1
			faces_clipped = 0
			# now we must remove all the faces that use these vertices
			vt_indexes = []
			for mtl in mesh.ordered_materials:
				new_faces = []
				for face in mtl.faces:
					keep_face = True
					for indices in face:
						if indices[0] not in clipped_v_indexes:
							new_index = index_map[indices[0]]
							#indices[0] = new_index
						else:
							keep_face = False
							faces_clipped += 1
							vt_indexes = indices[1]
							break
					if keep_face:
						new_faces.append(face)
				mtl.faces = new_faces
			#mesh.vertices = new_verts
			print "Clipped {0} faces".format(faces_clipped)

	def filter(self, scale=None, trans=None, clip=None, expand=None):
		"""
		Perform actions on a WaveFrontObj
		scale: tuple(x,y,z) or float
		trans: tuple(x,y,z)
		"""
		if type(scale) in [int, float]:
			scale = [scale, scale, scale]
		if type(trans) in [int, float]:
			trans = [trans, trans, trans]

		for v in self.vertices:
			if scale:
				v[0] = v[0] * scale[0]
				v[1] = v[1] * scale[1]
				v[2] = v[2] * scale[2]
			if trans:
				v[0] = v[0] + trans[0]
				v[1] = v[1] + trans[1]
				v[2] = v[2] + trans[2]

	def info(self):
		self.calculateDimensions()
		print "Wavefront OBJ - {0}".format(self.basename)
		if (self.mtllib):
			print "Wavefront MTL - {0}".format(self.mtllib.basename)
		print "\tVertices: {0}".format(self.total_vertices)
		print "\tTexture Coordinates: {0}".format(self.total_texture_coordinates)
		print "\tNormals: {0}".format(self.total_normals)
		print 'dimensions:'
		print "\twidth: {0}, height: {1}, depth: {2}".format(self.width, self.height, self.depth)
		print "\tX min: {0}\t\tmax: {1}".format(self.min_x, self.max_x)
		print "\tY min: {0}\t\tmax: {1}".format(self.min_y, self.max_y)
		print "\tZ min: {0}\t\tmax: {1}".format(self.min_z, self.max_z)
		for mesh in self.meshes.values():
			print "Mesh({0})".format(mesh.name)
			print "\tvertices: {0}".format(len(mesh.vertices))
			print "\ttexture coordinates: {0}".format(len(mesh.tex_coords))
			print "\tmaterials: {0}".format(len(mesh.ordered_materials))
			print "\ttextures: {0}".format(mesh.textureCount())
			for mtl in mesh.ordered_materials:
				print "\tMaterial({0})".format(mtl.name)
				print "\t\tfaces: {0}".format(len(mtl.faces))
				for t in mtl.textures:
					print "\t\tTexture: {0}".format(t)


	def merge(self, merge):
		merge_mtllibs = []
		for m in merge:
			if self.mtllib is None:
				if m.mtllib:
					self.mtllib = m.mtllib
				continue
			if m.mtllib:
				merge_mtllibs.append(m.mtllib)
			# next we merge the objects
			for mesh in m.ordered_meshes:
				print "merging {0} into {1}".format(mesh.name, self.basename)
				if mesh.name == "default" and len(mesh.vertices):
					mesh.name = m.basename
				self.ordered_meshes.append(mesh)
				self.meshes[mesh.name] = mesh

			self.vertices += m.vertices
			self.normals += m.normals
			self.tex_coords += m.tex_coords
		self.mtllib.merge(merge_mtllibs)

	def saveMeshes(self, f):
		nv = 0
		nvt = 0
		for mesh in self.ordered_meshes:
			print "writing mesh {0} - vertices({1}) faces({2})".format(mesh.name, mesh.faceCount(), len(mesh.vertices))
			for mat in mesh.ordered_materials:
				print "writing material: {0}".format(mat.name)
				f.write("usemtl {0}\n".format(mat.name))
				f.write("g {0}\n".format(mesh.name))
				for v in mat.faces:	
					f.write('f')
					rng = len(v)
					for n in range(rng):
						fv = int(v[n][0])+nv
						fuv = int(v[n][1])+nvt
						f.write(" {0}/{1}".format(fv, fuv))
					f.write("\n")
			nv += len(mesh.vertices)
			nvt += len(mesh.tex_coords)
		f.close()

	def saveVertices(self, f):
		count = 0
		for mesh in self.ordered_meshes:
			for v in mesh.vertices:
				count += 1
				f.write("v")
				for i in v:
					f.write(" {0}".format(i))
				f.write("\n")
		if count != len(self.vertices):
			print "WARNING: vertices with no faces: {0} dropped".format(len(self.vertices)-count)

	def saveNormals(self, f):
		for v in self.normals:
			f.write("vn")
			for i in v:
				f.write(" {0}".format(i))
			f.write("\n")

	def saveTextureCoords(self, f):
		for mesh in self.ordered_meshes:
			count = 0
			for v in mesh.tex_coords:
				count += 1
				f.write("vt")
				for i in v:
					f.write(" {0}".format(i))
				f.write("\n")		
		if count != len(self.tex_coords):
			print "WARNING: Texture Coordinates with no faces: {0} dropped".format(len(self.tex_coords)-count)

	def save(self, filename=None, merge=[]):
		"""
		Saves the file to disk
		"""
		outfile = filename
		if filename is None:
			outfile = self.filename

		name = os.path.basename(outfile).split('.')[0]

		newroot = os.path.dirname(os.path.abspath(outfile))
		if not os.path.exists(newroot):
			mkdir_p(newroot)

		#self.merge(merge)
		if self.mtllib is None:
			for mesh in self.ordered_meshes:
				if mesh.materialCount():
					for mat in mesh.ordered_materials:
						# we have material so we need a mtllib
						if self.mtllib is None:
							self.mtllib = MTLFile("temp.tml")
						self.mtllib.addMaterial(mat)

		if self.mtllib:
			print "saving mtllib {0}".format(name)
			self.mtllib.save(os.path.join(newroot, "{0}.mtl".format(name)))

		self.fullpath = os.path.abspath(outfile)
		self.filename = outfile
		self.root = os.path.dirname(self.fullpath)
		self.basename = os.path.basename(outfile)

		print("saving obj: '{0}'".format(outfile))
		f = open(outfile, 'w')
		f.write(OBJ_HEADER)
		f.write("\n")
		if self.mtllib:
			f.write("mtllib {0}\n".format(self.mtllib.name))

		self.saveVertices(f)
		self.saveNormals(f)
		self.saveTextureCoords(f)
		self.saveMeshes(f)
		f.close()











def main(opts, args):
	merge = []
	root_obj = None
	for f in args:
		obj = OBJFile(f)
		if root_obj is None:
			root_obj = obj
		else:
			merge.append(obj)
		obj.load()
	if len(merge):
		print "merging {0} objects".format(len(merge)+1)
		root_obj.merge(merge)
	if opts.verbose:
		root_obj.info()

	if opts.scale or opts.trans:
		print "applying filters to mesh"
		root_obj.filter(scale=opts.scale, trans=opts.trans)
		if opts.verbose:
			root_obj.info()

	if opts.clip_bottom:
		root_obj.clipBottom(opts.clip_bottom)

	if opts.clip_top:
		root_obj.clipTop(opts.clip_top)

	if opts.scale_bottom:
		root_obj.calculateDimensions()
		max_y = root_obj.min_y + (root_obj.height * opts.scale_bottom[0])
		ltf = (root_obj.min_x-1,max_y,root_obj.max_z+1)
		rbb = (root_obj.max_x+1,root_obj.min_y-1,root_obj.min_z-1)
		root_obj.filter_box(ltf, rbb, scale=opts.scale_bottom[1:])

	if opts.scale_top:
		root_obj.calculateDimensions()
		min_y = root_obj.max_y - (root_obj.height * opts.scale_top[0])
		ltf = (root_obj.min_x-100,root_obj.max_y+100,root_obj.max_z+100)
		rbb = (root_obj.max_x+100,min_y-100,root_obj.min_z-100)
		root_obj.filter_box(ltf, rbb, scale=opts.scale_top[1:])

	if opts.clip:
		print "clipping object"
		ltf = opts.clip[:3]
		rbb = opts.clip[3:6]
		root_obj.clip(ltf, rbb)

	if opts.extract:
		# pull the mesh out into a new object
		mesh = root_obj.getMesh(opts.extract)
		new_obj = OBJFile(opts.output)
		new_obj.addMesh(mesh)
		root_obj = new_obj

	if opts.output:
		root_obj.save(opts.output)

	if opts.view:
		os.system("open {0}".format(root_obj.filename))

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False )

	parser.add_option("--view", action="store_true", dest="view", default=False )	
	parser.add_option("-o", "--output", type="str", dest="output", default=None)
	parser.add_option("-e", "--extract", type="str", dest="extract", default=None)


	parser.add_option("--scale", type="float", dest="scale", default=None )
	parser.add_option("--trans", type="float", dest="trans", default=None )
	parser.add_option("--clip", type="float", nargs=6, dest="clip", default=None)
	parser.add_option("--clip-bottom", type="float", dest="clip_bottom", default=None)
	parser.add_option("--clip-top", type="float", dest="clip_top", default=None)

	parser.add_option("--scale-bottom", type="float", nargs=4, dest="scale_bottom", default=None)
	parser.add_option("--scale-top", type="float", nargs=4, dest="scale_top", default=None)


	(opts, args) = parser.parse_args()

	if len(args) == 0:   # if filename is not given
		parser.error('head file not given')

	main(opts, args)