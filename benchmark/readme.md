This benchmark folder consists of 2 folders

- for_testing_only. No need to look at it. It is just a playground for me to test adjust parameters, no systematic testing yet.
- result: relevant information about the benchmark of some example functions.

From now on, will only talk about 'result' folder

Here we test data with 3 ranges of values as follows.

- small: value 0.1 - 1, with precision at 0.001
- medium: value 1 - 100, with precision at 0.1
- large: value 100 - 1000, with precision at 1

And we test these values across, a dataset of 50, 100, 300, 600, and 1000 numbers

Now, how do we optimize the setting for each computation task. Looking at the setting file that ezkl generates, we can adjust 'scale', 'logrow', and 'mode' in the detail as follows.

- 'mode' can be either 'resources' or 'accuracy'. For practicality, we will always set it to 'resources' to optimize for the circuit size, and then adjust other parameters to address the accuracy issue instead.
- 'scale' is for representing float values either on inputs/outputs or the intermediate results during the computation graph that our zkstats want to compute. For example, if the lowest float number in those intermediates is around 0.001, the scale can be around 10, so that 0.001 \* 2^10 becomes integer.
  **We don't recommend to set it to default since it's mostly overkill.** Remember, the larger the scale is, the bigger size the circuit is. But sometimes the intermediate value in our computation requires that large value for example, if we need to calculate torch.log(), the floating point is very important, hence large scale.
- 'logrow' can be chosen arbitrarily between min(log_2(number of rows), log_2(lookup range)) to max(log_2(number of rows), log_2(lookup range)). The trade off is that if we choose lower logrow value , we will face with more "columns for non-linearity table".

  - **For now, based on our experiment, the best value to select logrow is to select the lowest logrow in the range such that ceiling(lookup_range/2^logrow) <=5**
