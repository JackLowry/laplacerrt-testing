import math
import pyamg
import numpy as np
import skfem as fem
from skfem.models.poisson import laplace
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
im = Image.open("world3.png")
image = np.asarray(im)[:, :, 0]/255

img_x, img_y = image.shape

image[0, :] = 0
image[-1, :] = 0
image[:, 0] = 0
image[:, -1] = 0

image = image.flatten()
boundary_indices_flattened = np.arange(image.shape[0])
boundary_indices_flattened = boundary_indices_flattened[image == 0]
boundary_values_flattened = np.ones_like(boundary_indices_flattened)

start_idx = 30*img_y + 20
end_idx = img_x*350 + 650

boundary_indices_flattened = np.append(boundary_indices_flattened, [start_idx])
boundary_indices_flattened = np.append(boundary_indices_flattened, [end_idx])

boundary_values_flattened = np.append(boundary_values_flattened, [0])
boundary_values_flattened = np.append(boundary_values_flattened, [1])

free_indices_flattened = np.arange(image.shape[0])
free_indices_flattened = free_indices_flattened[image == 1]

free_indices_flattened = free_indices_flattened[free_indices_flattened != start_idx]
free_indices_flattened = free_indices_flattened[free_indices_flattened != end_idx]

A = pyamg.gallery.poisson((img_x,img_y), format='csr')  # 2D Poisson problem on 500x500 grid
b = np.zeros(A.shape[0])                      # pick a random right hand side

x = np.zeros(A.shape[0])
x[boundary_indices_flattened] = boundary_values_flattened

# probably a type mismatch here -- maybe not?
AII, bI, xI, I = fem.condense(A, b, D=boundary_indices_flattened, x=x)

ml = pyamg.ruge_stuben_solver(AII)                    # construct the multigrid hierarchy
print(ml)                                            # print hierarchy information
x = ml.solve(bI, tol=1e-60, maxiter=500)                          # solve Ax=b to a tolerance of 1e-10
print("residual: ", np.linalg.norm(bI-AII*x))          # compute norm of residual vector

image_solved = image.copy()
image_solved[free_indices_flattened] = x

image_solved[start_idx] = 0
image_solved[end_idx] = 1

image_solved = np.reshape(image_solved, (img_x, img_y))

display_image = np.log(image_solved.copy())

x = 350
y = 650

nodes = []
limit = 0


figure, axis = plt.subplots(figsize=(7.6, 6.1))
axis.imshow(image_solved)
plt.savefig("solved.png")

# plt.ion()
# plt.show(block=False)

for char in (pbar := tqdm(range(1000000000000))):
    pbar.set_description(f"pos: {x}, {y}")
    if limit > 100000000:
        break

    gradx = image_solved[int(x+1)][int(y)] - image_solved[int(x-1)][int(y)]
    grady = image_solved[int(x)][int(y+1)] - image_solved[int(x)][int(y-1)]

    maggrad = math.sqrt(gradx**2 + grady**2)

    if(maggrad != 0):
        a = 3 / maggrad
        x = x-a*gradx
        y = y-a*grady
        
    # display_image[round(x), round(y)] = 10

    # axis.imshow(display_image)
    # plt.gcf().canvas.draw()
    # figure.colorbar()
