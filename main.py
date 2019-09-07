from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.config import Config
Config.set('kivy', 'log_level', 'debug')
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stencilview import StencilView
from kivy.uix.screenmanager import ScreenManager, Screen

from orthographic import *

class BoxStencil( FloatLayout, StencilView ):
	pass

class OrthographicScreen( Screen ):
	main_area        = ObjectProperty( )
	left_area        = ObjectProperty( )
	right_nav_bar    = ObjectProperty( )
	structure_list   = ObjectProperty( None, allownone = True )
	background_color = ObjectProperty( "black" )
	ortho            = ObjectProperty( None, allownone = True )

	def on_enter( self ):
		self.setup_ortho( )

	def setup_ortho( self ):
		self.ortho = OrthographicRenderer( root   = self,
				                   screen = self,
				                   size   = self.left_area.size,
						   pos    = self.left_area.pos )
		self.left_area.add_widget( self.ortho )
		self.ortho.setup_scene( )
				                 

class TestApp( App ):
	def build( self ):
		self.sm = ScreenManager( )
		self.sm.add_widget( OrthographicScreen( name = "orthographic" ) )
		return self.sm

if __name__ == '__main__':
	root = TestApp( )
	root.run( )


