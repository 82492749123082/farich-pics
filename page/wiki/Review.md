# Review of basic _trackless_ ring recognition methods
    
## Methods

### *Hough transformation* [1, 3]

#### Description

* Advantages
    - has implementation in Python
* Disadvantages
    - multidimensional features space. Dimensional decreasing 
    - slow
    - performs well only on noiseless data   


### *ElasticNet* [2]

#### Description
The authors thought out the energy function that we minimize and get rings parameters

* Advantages
    - very good efficiency
    - extremely high speed 1 ms
* Disadvantages
    - uses iterations
    * **we cannot get the same results as in the article [2]**

### *Almagest* [4]

#### Description
The authors use Ptolemy geometry theorem related with circles to check points sets

* Advantages
    * fast? multiple ring searching and can be used in trigger
    * GPU support
* Disadvantages
    * `O(N) ~ N!/( (N-4)! 4! )` for circles without tricks
    * circles (i've used pascal theorem for ellipses)

### Articles
1. [Ring recognition and electron identification in the RICH](https://doi.org/10.1088/1742-6596/219/3/032015) (Lebedev, Höhne, Ososkov)
1. [Elastic net for stand-alone RICH ring finding](https://doi.org/10.1016/j.nima.2005.11.215) (Gorbunov, Kisel; 2006)
1. [In search of the rings: Approaches to Cherenkov ring finding and reconstruction in high energy physics](https://doi.org/10.1016/j.nima.2008.07.066) (Wilkinson; 2008)
1. [Almagest, a new trackless ring finding algorithm](https://doi.org/10.1016/j.nima.2014.05.073) (Lamanna; 2014)