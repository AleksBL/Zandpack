- The fitted function should also have coefficients that exceed the magnitude
  of the sampled functions by orders of magnitude (specific numerical tolerance
  in implementation). This is to avoid two closely spaced poles with different
  but large (in magnitude) cancelling each other and leading to numerical
  trouble later in the pipeline.  
