from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.graphics import *
from kivy.core.text import Label as CoreLabel
from kivy.graphics.opengl import *
from kivy.properties import ObjectProperty
from kivy.clock import Clock
import numpy as np

from scipy.linalg import norm

class OrthographicRenderer( Widget ):
	SCALE_FACTOR = 0.05
	MAX_SCALE = 5.0
	MIN_SCALE = 0.3
	ROTATE_SPEED = 1.

	POLAR_RADIUS = 4.0
	STEP = 1

	root               = ObjectProperty( )
	screen             = ObjectProperty( )

	borehole_size      = ObjectProperty( 0 )
	borehole_pos       = ObjectProperty( 0 )
	structures         = ObjectProperty( None )
	borehole_segments  = ObjectProperty( None )
	background_color   = ObjectProperty( "black" )

	start_position     = ObjectProperty( None )

	current_depth      = ObjectProperty( 0 )

	def __init__( self, **kwargs ):
		print( "OrthographicRenderer: __init__: start" )

		# Load the shaders
		kwargs[ 'shader_file' ] = 'shaders.glsl'
		self.canvas = RenderContext( compute_normal_mat = True )

		shader_file = kwargs.pop( 'shader_file' )
		self.canvas.shader.source = resource_find( shader_file )
		self._touches = [ ]

		# Setup the vertex format, to be used with the shader
		self.vertex_format = [ ( b'v_pos',    3, 'float' ),
				       ( b'v_color', 4, 'float' ),
				       ( b'v_tc0',    2, 'float' ) ]

		# Setup keyboard event handler
		Window.request_keyboard( None, self ).bind( on_key_down = self.on_keyboard_down )

		# Call super
		super( OrthographicRenderer, self ).__init__( **kwargs )

	# called whenever the widgets size changes
	def on_size( self, instance, value ):
		self.update_projection( )

	def update_glsl( self, *args ):
		asp = self.width / float( self.height )
		proj = Matrix( ).view_clip( -asp, asp, -1, 1, 1, 100, 1 )
		self.canvas[ 'projection_mat' ] = proj

	def calculate_borehole_segments( self ):
		color = [ 0.50, 0.50, 0.50, 1, 0, 0 ]
		self.borehole_size = int( 10 )

		self.borehole_start = 0
		self.borehole_end = self.borehole_size

		distance = int( self.borehole_end - self.borehole_start )
		self.points = [
			[ 0., 0.,         0.,  0.,         0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 1., 0.99999845, 0., -0.00174533, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 2., 1.9999969,  0., -0.00349066, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 3., 2.9999955,  0., -0.00523598, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 4., 3.9999938,  0., -0.00698131, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 5., 4.9999924,  0., -0.00872664, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 6., 5.999991,   0., -0.01047197, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 7., 6.9999895,  0., -0.0122173,  0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 8., 7.9999876,  0., -0.01396263, 0.5, 0.5, 0.5, 1., 0., 0. ],
			[ 9., 8.999987,   0., -0.01570795, 0.5, 0.5, 0.5, 1., 0., 0. ],
		]

		self.borehole_segments = np.asarray( self.points, 'f' )

	def calculate_structures( self ):

		self.structures = [ ]

		self.alpha = 1.0

		points1 = [ [ 7.307719,  1.211168,  -0.48409978, 1., 0.19607843, 0.8, self.alpha, 0., 0., ],
		 	    [ 6.733826, -0.7047252, -0.48409978, 1., 0.19607843, 0.8, self.alpha, 0., 0., ],
			    [ 8.424542, -1.211168,   0.45664176, 1., 0.19607843, 0.8, self.alpha, 0., 0., ],
			    [ 8.998436,  0.7047252,  0.45664176, 1., 0.19607843, 0.8, self.alpha, 0., 0., ] ]

		points2 = [ [ 7.6652303,  0.61850137,  -0.51519805, 1., 0.19607843, 0.8, self.alpha, 0., 0., ], 
			    [ 8.389436,  -1.2457747,   -0.51519805, 1., 0.19607843, 0.8, self.alpha, 0., 0., ], 
			    [10.004185,  -0.61850137,   0.48435906, 1., 0.19607843, 0.8, self.alpha, 0., 0., ],
			    [ 9.279979,   1.2457747,    0.48435906, 1., 0.19607843, 0.8, self.alpha, 0., 0., ] ]


		indices = np.asarray( [ 0, 1, 2, 0, 2, 3], 'd' )
		vertices1 = np.asarray( points1, 'f' )
		vertices2 = np.asarray( points2, 'f' )


		plane_01 = { 
			"vertices": vertices1,
			"indices": indices,
			"mesh": None,
		}

		plane_02 = {
			"vertices": vertices2,
			"indices": indices,
			"mesh": None,
		}

		self.structures.append( plane_01 )
		self.structures.append( plane_02 )

	def update_projection( self, *args ):
		asp = float( Window.width ) / Window.height / 2.0
		offset = ( self.center_x / ( Window.width / 2 ) ) * 0.25

		left = -asp + offset
		right = asp + offset
		bottom = -0.5
		top = 0.5

		proj = Matrix( ).view_clip( left, right, bottom, top, 1, 100, 1 )
		self.canvas[ 'projection_mat' ] = proj

	def setup_gl_context(self, *args):
		glEnable( GL_DEPTH_TEST )

	def reset_gl_context(self, *args):
		glDisable( GL_DEPTH_TEST )

	def setup_scene( self ):
		print( "OrthographicRenderer: setup_scene:" )

		self.calculate_borehole_segments( )

		self.calculate_structures( )

		with self.canvas:
			# Setup Depth checking
			self.cb = Callback(self.setup_gl_context)

			self.translate = Translate( 0, 0, -15 )

			self.rot = Rotate( 0, 1, 1, 1 )
			self.rotx = Rotate( 0, 1, 0, 0 )
			self.roty = Rotate( 0, 0, 1, 0 )

			self.scale = Scale( 1.0 )

			if self.borehole_segments is not None:
				PushMatrix( )
				self.borehole_translate = Translate( 0, 0, 0 )

				for i in self.structures:
					print( "v: {}".format( i[ "vertices" ].ravel( ) ) )
					print( "i: {}".format( i[ "indices"  ].ravel( ) ) )

					i[ "mesh" ] = Mesh( vertices = i[ "vertices" ].ravel( ),
					                    indices  = i[ "indices" ].ravel( ),
					                    fmt = self.vertex_format,
					                    mode = "triangles" )

				for i in range( 0, len( self.borehole_segments ) - 1 ):
					self.create_main_cylinder( self.borehole_segments[ i ],
							           self.borehole_segments[ i + 1 ],
								   self.borehole_segments[ i ][ 0 ] )


				PopMatrix( )

				PushMatrix( )
				self.angle_translate = Translate( 0, 0, 0 )
				self.angle_roty = Rotate( 0, 0, 1, 0 )
				PopMatrix( )

			self.cb = Callback(self.reset_gl_context)

	def create_main_cylinder( self, point_one, point_two, depth ):
		p0 = point_one[ 1:4 ]
		p1 = point_two[ 1:4 ]

		# Radius
		R = 0.1

		# Vector in direction of axis
		v = p1 - p0

		print( "p0: {}".format( p0 ) )
		print( "p1: {}".format( p1 ) )
		print( "v:  {}\n".format( v ) )

		# Check if v = [ 0.0, 0.0, 0.0 ]
		if not any( v ):
			v[ 0 ] = 0.000001

		# Find the magnitude of the vector
		mag = norm( v )

		# Unit vector in the direction of axis
		v = v / mag

		# Make a vector not in the same direction as v
		not_v = np.array( [ 1, 0, 0 ] )
		if ( v == not_v ).all( ):
			not_v = np.array( [ 0, 1, 0 ] )

		# Make vector perpendicular to v
		n1 = np.cross( v, not_v )

		# Normalize n1
		n1 /= norm( n1 )

		# Make unit vector perpendicular to v and n1
		n2 = np.cross( v, n1 )

		# Surface ranges over 0 to 2 * pi
		size = 13
		theta = np.linspace( 0, 2 * np.pi, size ).reshape( size, 1 )

		x1 = p0[ 0 ] + R * np.sin( theta ) * n1[ 0 ] + R * np.cos( theta ) * n2[ 0 ]
		y1 = p0[ 1 ] + R * np.sin( theta ) * n1[ 1 ] + R * np.cos( theta ) * n2[ 1 ]
		z1 = p0[ 2 ] + R * np.sin( theta ) * n1[ 2 ] + R * np.cos( theta ) * n2[ 2 ]

		x2 = p1[ 0 ] + R * np.sin( theta ) * n1[ 0 ] + R * np.cos( theta ) * n2[ 0 ]
		y2 = p1[ 1 ] + R * np.sin( theta ) * n1[ 1 ] + R * np.cos( theta ) * n2[ 1 ]
		z2 = p1[ 2 ] + R * np.sin( theta ) * n1[ 2 ] + R * np.cos( theta ) * n2[ 2 ]

		# Vertice format: [ x, y, z, r, g, b, a, u, v ]
		# Create a circle at either side of the cylinder, corresponding to the two points
		vertices_one = np.zeros( ( size, 9 ), 'f' )
		vertices_two = np.zeros( ( size, 9 ), 'f' )

		# Create the surface of the cylinder
		vertices_three = np.zeros( ( size * 6, 9 ), 'f' )
		indices_three  = np.zeros( ( 1, size * 6 ), 'f' )

		color1 = np.ones( ( size, 6 ), 'f' )
		color2 = np.ones( ( size, 6 ), 'f' )
		color1 *= point_one[ 4:10 ]
		color2 *= point_two[ 4:10 ]

		vertices_one = np.hstack( ( x1, y1, z1, color1 ) )
		vertices_two = np.hstack( ( x2, y2, z2, color2 ) )

		for i in range( 0, size - 1 ):
			j = i * 6
			verts = [ vertices_one[ i ],
			          vertices_two[ i ],
				  vertices_two[ i + 1 ],
				  vertices_one[ i ],
				  vertices_one[ i + 1 ],
				  vertices_two[ i + 1] ]
			vertices_three[ j:j + 6 ] = verts
			indices_three[ 0,j:j + 6 ] = np.arange( j, j + 6 )

		with self.canvas:
			self.cb = Callback(self.setup_gl_context)
			# Draw the cylinder mesh
			Mesh( vertices = vertices_three.ravel( ),
			      indices  = indices_three.ravel( ),
			      fmt      = self.vertex_format,
			      mode     = 'triangles' )

			self.cb = Callback(self.reset_gl_context)

	def on_keyboard_down( self, keyboard, keycode, text, modifiers ):

		if keycode[ 1 ] == "escape":
			self.roty.angle = 0
			self.rotx.angle = 0
			self.angle_roty.angle = 0
			self.scale.xyz = ( 1.0, 1.0, 1.0 )

		## This shifts the line
		if keycode[ 1 ] == "right":

			if self.borehole_segments is None:
				return

			if self.current_depth >= len( self.borehole_segments ) - 1:
				return
			
			current_pos = self.borehole_segments[ self.current_depth ]
			next_pos    = self.borehole_segments[ self.current_depth + 1 ]

			self.current_depth += 1

			diff = next_pos - current_pos
			self.borehole_translate.xyz -= diff[ 1:4 ] 

		elif keycode[ 1 ] == "left":
			if self.borehole_segments is None:
				return

			if self.current_depth <= 0:
				return

			current_pos = self.borehole_segments[ self.current_depth ]
			next_pos    = self.borehole_segments[ self.current_depth - 1 ]

			self.current_depth -= 1

			diff = current_pos - next_pos
			self.borehole_translate.xyz += diff[ 1:4 ]

		elif keycode[ 1 ] == "down":
			self.alpha -= 0.1
			if self.alpha < 0.0:
				self.alpha = 0.0

			for i in self.structures:
				i[ "vertices" ][ :,6:7 ] = self.alpha
				i[ "mesh" ].vertices = i[ "vertices" ].ravel( )

		elif keycode[ 1 ] == "up":
			self.alpha += 0.1
			if self.alpha > 1.0:
				self.alpha = 1.0

			for i in self.structures:
				i[ "vertices" ][ :,6:7 ] = self.alpha
				i[ "mesh" ].vertices = i[ "vertices" ].ravel( )

	def ignore_undertouch( func ):
		def wrap( self, touch ):
			gl = touch.grab_list
			if len( gl ) == 0 or ( self is gl[ 0 ]( ) ):
				return func( self, touch )
		return wrap

	@ignore_undertouch
	def on_touch_down( self, touch ):
		if self.screen.right_nav_bar.collide_point( *touch.pos ):
			return

		touch.grab( self )
		self._touches.append( touch )

		if touch.is_double_tap:
			self.roty.angle = 0
			self.rotx.angle = 0
			self.angle_roty.angle = 0
			self.scale.xyz = ( 1.0, 1.0, 1.0 )

		if 'button' in touch.profile and touch.button in ( 'scrollup', 'scrolldown' ):

			if touch.button == "scrolldown":
				scale = self.SCALE_FACTOR

			if touch.button == "scrollup":
				scale = -self.SCALE_FACTOR

			xyz = self.scale.xyz
			scale = xyz[ 0 ] + scale
			if scale < self.MAX_SCALE and scale > self.MIN_SCALE:
				self.scale.xyz = ( scale, scale, scale )

	@ignore_undertouch
	def on_touch_up( self, touch ):
		touch.ungrab( self )
		if touch in self._touches:
			self._touches.remove( touch )

	def define_rotate_angle( self, touch ):
		x_angle = ( touch.dx / self.width ) * 360.0  * self.ROTATE_SPEED
		y_angle = -1 * ( touch.dy / self.height ) * 360.0 * self.ROTATE_SPEED

		return x_angle, y_angle

	@ignore_undertouch
	def on_touch_move( self, touch ):
		if touch in self._touches and touch.grab_current == self:
			if len( self._touches ) == 1:
			# here do just rotation        
				ax, ay = self.define_rotate_angle( touch )

				self.roty.angle += ax
				self.rotx.angle += ay
				self.angle_roty.angle -= ax
