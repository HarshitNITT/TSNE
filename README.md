# Why Multiple Map Tsne
For visualising non metric data i.e the data which does not follow one of the following metric axioms:
```
d(A,B)≥0,
d(A,B)=0  iff   A=B,
d(A,B)=d(B,A),
d(A,C)≤d(A,B)+d(B,C)
```
## Single map Tsne represent metric data which has 2 Limitations:
```
**The triangle inequality that holds in metric spaces induces transitivity of similarities**
Which means that if A and B are closer to each other and B and C are closer to each other the by Triangle inequality A and C are closer to each other.
e.g Consider the word tie, which has a semantic similarity to words such as suit and tuxedo. In a low-dimensional metric map of the input objects, these three words need to be close to each other. However, the word tie is ambiguous: it is also semantically similar to words such as rope and knot , and should therefore be close to these words as well.
###The number of points that can have the same point as their nearest neighbor is limited.
If we represent data in a single map it will be only one 2-d graph.At most five points can have the same point as their nearest neighbor(by arranging them in a pentagon that is centered on the point).So if we have more than objects as nearest neighbour of the points then we cannot represent them.  As a result, a low-dimensional metric map constructed by multidimensional scaling cannot faithfully visualize the large number  of similarities of “central” objects with other objects.
```
## How Does Multiple Maps TSNE Solve This Problem
```
Multiple Map Tsne Represent points in the maps as objects that have importance  weights in each of these maps which represent the importance in each of the maps,the similarity between the two is the summation of similarity between them  over all the maps. The similarity in each of the maps is related to the importance weights and their proximity. If the 2 points are closer and their importance weight is high in one of the maps then they are similar.
For example, the word tie can be close to tuxedo in a map in which knot has a low weight, and close to knot in another map in which tuxedo a low weight. This captures the similarity of tie to both tuxedo and knot without forcing tuxedo to be close to knot.
This solves the problem of non metric spaces.
```
## Mathematics Behind the Single map TSNE
```
Given n high Dimensional vectors  x 1 , … , x N  t-SNE first computes probabilities p i j {\displaystyle p_{ij}} p_{ij} that are proportional to the similarity of objects x i {\displaystyle \mathbf {x} _{i}} \mathbf {x} _{i} and x j {\displaystyle \mathbf {x} _{j}} \mathbf {x} _{j}, as follows:
[]("https://wikimedia.org/api/rest_v1/media/math/render/svg/2cc3ef3b4d237787cd82e5ef638d96d642a1e43d")
The similarity of datapoint x j {\displaystyle x_{j}} x_{j} to datapoint x i {\displaystyle x_{i}} x_{i} is the conditional probability, p j | i {\displaystyle p_{j|i}} {\displaystyle p_{j|i}}, that x i {\displaystyle x_{i}} x_{i} would pick x j {\displaystyle x_{j}} x_{j} as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at x i.
Since we are only interested in pairwise similarities between points, t-SNE sets p ii=0.
```
> ![](http://www.sciweavers.org/upload/Tex2Img_1503836598/render.png)

> ![](http://www.sciweavers.org/upload/Tex2Img_1503836873/render.png)

> ![](http://www.sciweavers.org/upload/Tex2Img_1503836941/render.png)
```
The aim of Tsne is to model high dimensional vectors into low dimensional vectors such that similarity between two points qij which represent similarity in the low dimensional space of the counterparts yi and yj in low dimwnsional space. The error between the input similarities pij and their counterparts in the low-dimensional map q ij is measured by means of the Kullback-Leibler divergence between the distributions.

We need to minimize kullback divergance in order to have qij value similar to that of pij so that we can attain the internal structure of the map which is implemented using gradient descent
```
> ![](http://www.sciweavers.org/upload/Tex2Img_1503832642/render.png)

> ![](http://www.sciweavers.org/upload/Tex2Img_1503837861/render.png)

## Mathematics Behind Multiple map Tsne
```
Multiple maps t-SNE constructs a collection of M maps, all of which contain N points (one for each of the N input objects). In each map with index m , a point with index i has a so-called importance weight π(m)i that measures the importance of point i in map m. Because of the probabilistic interpretation of our model, our weight function π(m)i must be positive for all i and m.
for all m for a particular i the π(m)i summation must be 1.

So we redefine qij as follows:
```

> ![](http://www.sciweavers.org/upload/Tex2Img_1503840704/render.png)
```
Because we require the importance weights π(m)i to be positive and we require the importance weights π (m) i for a single point i to sum up to 1 over all maps, direct optimization of the cost function w.r.t. the parameters π (m)i is tedious. To circumvent this problem, we represent the importance weights π(m)i in terms of unconstrained weight w(m)i (using an idea that is similar to that of softmax units) as follows:
```
> ![](http://www.sciweavers.org/upload/Tex2Img_1503841227/render.png)


# Gradient Descent
```
Gradient descent is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function. This iterative minimization is achieved using calculus, taking steps in the negative direction of the function gradient.
``` 
## Applying gradient descent in Linear Regression
```
Simply speaking in Linear Regression we have to fit a line which best represent data .
It is Linear Regression so line is y=mx+b where m is the slope and b is the y-intercept.
``` 
Consider the following Example:
![](https://spin.atomicobject.com/wp-content/uploads/points_for_linear_regression1.png)

What we do is fit a arbitrary line with m and b values and calculate the error function as follows:
![](http://www.sciweavers.org/upload/Tex2Img_1503843474/render.png)

Code Snippet Calculating the error Function is given as:
```python 
def computeErrorForLineGivenPoints(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        totalError += (points[i].y - (m * points[i].x + b)) ** 2
    return totalError / float(len(points))
```
Suppose we take an arbitrary value of m and b  and we calculate the value of error function as arbitrary. We have to minimize this error function so that we get particular value of m and b for which line fits the best. we Calculate this m and b by gradient Descent.
The Error function would look something like That:

![](https://spin.atomicobject.com/wp-content/uploads/gradient_descent_error_surface.png)

We will take arbitrary value of m,b and move downhill to the global minima.
```
To run gradient descent on this error function, we first need to compute its gradient. The gradient will act like a compass and always point us downhill. To compute it, we will need to differentiate our error function. Since our function is defined by two parameters (m and b), we will need to compute a partial derivative for each. These derivatives work out to be:
```
![](https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png)
We can initialize our search to start at any pair of m and b values (i.e., any line) and let the gradient descent algorithm march downhill on our error function towards the best line. Each iteration will update m and b to a line that yields slightly lower error than the previous iteration.
```python
def stepGradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        b_gradient += -(2/N) * (points[i].y - ((m_current*points[i].x) + b_current))
        m_gradient += -(2/N) * points[i].x * (points[i].y - ((m_current * points[i].x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]
```
```
The learningRate variable controls how large of a step we take downhill during each iteration. If we take too large of a step, we may step over the minimum. However, if we take small steps, it will require many iterations to arrive at the minimum.
```
Here Are the few snapshots of the gradient Descent Algorithm in action:



![](https://spin.atomicobject.com/wp-content/uploads/gradient_descent_search1.png)
