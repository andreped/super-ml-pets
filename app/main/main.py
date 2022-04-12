import kivy
from kivy.app import App
from kivy.uix.button import Label

class Main(App):
    def build(self):
        return Label(text ="Hello Geeks")
  
  

if __name__ == "__main__":
    m = Main()
    m.run()