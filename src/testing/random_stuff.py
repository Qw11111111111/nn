from src.layers import LinearLayer


class Model:
    
    def __init__(self) -> None:
        self.layers = [
            LinearLayer((5, 2))
        ]
        
    def print(self):
        for row in self.layers[0].weights:
            print(*row)

model = Model()

class Optimizer:
    
    def __init__(self, model: Model):
        self.model = model
    
    def step(self):
        self.model.layers[0].weights += 1


optimizer = Optimizer(model)

model.print()


for i in range(10):
    optimizer.step()
    
print("\npost\n")
model.print()
    

    