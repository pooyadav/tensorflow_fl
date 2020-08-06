import torch as torch
import syft as sy
hook = sy.TorchHook(torch)



jake = sy.VirtualWorker(hook, id="jake")
print("Jake has: " + str(jake._objects))
x = torch.tensor([1, 2, 3, 4, 5])
x = x.send(jake)
print("x: " + str(x))
print("Jake has: " + str(jake._objects))

x = x.get()
print("x: " + str(x))
print("Jake has: " + str(jake._objects))

john = sy.VirtualWorker(hook, id="john")
x = x.send(jake)
x = x.send(john)
print("x: " + str(x))
print("John has: " + str(john._objects))
print("Jake has: " + str(jake._objects))

jake.clear_objects()
john.clear_objects()
print("Jake has: " + str(jake._objects))
print("John has: " + str(john._objects))

y = torch.tensor([6, 7, 8, 9, 10]).send(jake)
y = y.move(john)
print(y)
print("Jake has: " + str(jake._objects))
print("John has: " + str(john._objects))