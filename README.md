# Mandelbrot Set CUDA

A Mandelbrot Set implementation using Nvidia CUDA library, GPU multi-core processing .Mandelbrot set has been named and discovered by Benoit B. Mandelbrot (Horgan, 2009) Firstly, we will start explaining by what Set is. A set is a collection of objects. For example, a set can be representing as a list of numbers between two specified number values. A characteristic of sets is that is not necessary to specify the length of it as the lists do, they might have arbitrary length and an infinite collection. If we can imagine it differently, a set might look like a function that has an input one object and returns a boolean (“true or false”) if the object belongs to the collection of that set. Therefore, the Mandelbrot Set is a collection of complex numbers.  The mathematical equation that define the Mandelbrot set is F(z) = z2 + c where c is a complex number and z is a number that starts from 0. This equation runs into a loop by specified number of iterations or ends on a threshold. Firstly, the z value becomes the c as z starts from 0 (Weisstein's, n.d.). C is a complex number that means it holds an imaginary unit and a real number part and can be written as a + b*i where a and b are real numbers and i is an imaginary unit. Imaginary number we call what it can not be measured by real numbers like x2 = -1 where is impossible to solve this equation without using an imaginary unit for example i = √-1 (Khan Academy, n.d.). The set holds the c values that are running into the loop and are not tending to diverge. For example, if we set the value of c = 0 then f(z) = z2 + 0 is returning true because we will return to the same function as we started thus, we don’t have any diverge. The equation of the set and what might return as a belonging complex number depends on the number of the iterations that we specified, as we increase the number iterations, we find more accurate results as what the Mandelbrot set describes (Weisstein's, n.d.). If we shrink the volume of this set by specifying dimensions, height that will represent the complex numbers values and width which will be represent the real numbers, we will end up having a 2d graph that will represent the collection of the Mandelbrot. We then can generate an image on the Mandelbrot set by selecting the complex numbers that belongs to it and change the colours of each data point on the axis. A significant behaviour that Mandelbrot set image seems to present is that is an abstract fractal. Fractal it is figures that follow similar and infinitely complex patterns which means as we zoom in into the content of the Mandelbrot set image that we generated, we will see the same or similar figure patterns
