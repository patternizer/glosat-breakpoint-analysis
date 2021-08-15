![image](https://github.com/patternizer/glosat-breakpoint-analysis/blob/main/103810-cusum-curve-linear-tree-loop.png)
![image](https://github.com/patternizer/glosat-breakpoint-analysis/blob/main/685880-cusum-curve-linear-tree-loop.png)

# glosat-breakpoint-analysis

Python algorithm to calculate breakpoints in local expectation Krigining output CUSUM timeseries. Part of ongoing work on land surface air temperature station homogensiation as part of the project [Glosat](www.glosat.org) 

## Contents

* `piecewise_linear_spline_regression.py` - python code to apply piecewise linear spline regression (PWLSR) to CUSUM timeseries from the local expectation Kriging homogenisation
* `basis_expansions.py` - fork of piecewise linear spline regressor by [Mathew Drury](https://github.com/madrury/basis-expansions)
* `linear_tree_regression.py` - changepoint detection code to calculate breakpoints in the LEK CUSUM timeseries using linear tree regression (LTR) built using the linear tree regression driver by [Marco Cerliani](https://github.com/cerlymarco/linear-tree)

The first step is to clone the latest glosat-breakpoint-analysis code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-breakpoint-analysis.git
    $ cd glosat-breakpoint-analysis

### Usage

The code was tested locally in a Python 3.8.11 virtual environment.

    $ python piecewise_linear_spline_regression.py
    $ python linear_tree_regression.py

You will need to install the LTR driver with pip install linear-tree. Check other python library dependencies in the code header.
Both codes take as input the CSV station local expectation Kriging CUSUM timeseries output by the homogenisation code. Examples for testing in DATA/
The LTR code output the breakpoints in decimal year format.

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


