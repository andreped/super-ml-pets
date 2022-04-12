from kivy.config import Config
Config.set('graphics', 'width', '500')
Config.set('graphics', 'height', '500')
Config.set('graphics', 'resizable', False)

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
#from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.core.image import Image
#from kivy.graphics import BorderImage
from kivy.graphics import Color, Rectangle, Mesh, Point, Ellipse
#from kivy.uix.image import AsyncImage
from kivy_garden.graph import Graph, MeshLinePlot
from math import sin

class NeuralNetworkPlot(MeshLinePlot):
    def _set_mode(self, value):
        if hasattr(self, '_mesh'):
            self._mesh.mode = value

    def create_drawings(self):
        self._color = Color(*self.color)
        self._mesh = Mesh(mode='line_strip')
        self.bind(
            color=lambda instr, value: setattr(self._color, "rgba", value))
        return [self._color, self._mesh]

    def draw(self, *args):
        super(MeshLinePlot, self).draw(*args)
        self.plot_mesh()

    def plot_mesh(self):
        points = [p for p in self.iterate_points()]
        mesh, vert, _ = self.set_mesh_size(len(points))
        for k, (x, y) in enumerate(points):
            vert[k * 4] = x
            vert[k * 4 + 1] = y
        mesh.vertices = vert

    def set_mesh_size(self, size):
        mesh = self._mesh
        vert = mesh.vertices
        ind = mesh.indices
        diff = size - len(vert) // 4
        if diff < 0:
            del vert[4 * size:]
            del ind[size:]
        elif diff > 0:
            ind.extend(range(len(ind), len(ind) + diff))
            vert.extend([0] * (diff * 4))
        mesh.vertices = vert
        return mesh, vert, ind

class AIVisualize(FloatLayout):
    def visualize(self):
        graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
        x_ticks_major=25, y_ticks_major=1,
        y_grid_label=True, x_grid_label=True, padding=5,
        x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1)
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
        graph.add_plot(plot)

class Root(FloatLayout):
    def start_ai(self):
        print ("!FIOJASD")

class Main(App):
    def build(self):
        return Root()
  
  

if __name__ == "__main__":
    m = Main()
    m.run()