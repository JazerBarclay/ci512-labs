class Resteraunt:

    def __init__(self, name, cuisine):
        self.name = name
        self.cuisine = cuisine
    
    def describe_restaurant(self):
        print(self.name + " makes " + self.cuisine + " food!")