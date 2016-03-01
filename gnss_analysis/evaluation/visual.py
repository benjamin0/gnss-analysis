import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(metrics, metric_name, metric_units, axis=None,
                *args, **kwdargs):
  """
  Creates a single plot for a given metric.  The columns
  will become independent lines, and the y axis will be labled
  using the metric name and units.
  """
  if axis is None:
    axis = plt.gca()
  # TODO: for now we only show the epoch count, it'd be nice to have
  #   an actual time axis, but resizing them isn't straight forward.
  epochs = np.arange(metrics.index.size)
  lines = axis.plot(epochs, metrics.values, *args, **kwdargs)
  plt.legend(metrics.columns)
  axis.set_ylabel("%s (%s)" % (metric_name, metric_units))
  axis.set_xlabel('Epoch')
  return lines


def plot_metric_iterative(iter_metric, metric_name, metric_units,
                          draw_every=10, ax=None):
  """
  Takes an iterable of summary metrics and produces an incrementally
  updating plot showing the metric as it evolves
  """
  if ax is None:
    ax = plt.gca()

  # we end up iterating through solutions and plotting the results
  # as they come in.  metric_buffer aggregates all the values fo
  # the metric to aid with plotting.
  epoch_buffer = []
  metric_buffer = pd.DataFrame()

  # iterate over epochs, producing the solution set for each
  # filter at the given epoch.
  for i, metric in enumerate(iter_metric):

    if np.all(np.isfinite(metric)):
      epoch_buffer.append(i)
      # add to the buffer
      metric_buffer = pd.concat([metric_buffer, metric])

      if len(epoch_buffer) == 1:
        # initialize the plot if this is the first iteration
        plt.ion()
        lines = plot_metric(metric_buffer, metric_name, metric_units, axis=ax)
        plt.show()
        plt.pause(0.001)
      else:
        # otherwise update the plot by first resetting the data of the line plots
        for line, vals in zip(lines, metric_buffer.values.T):
          line.set_data(epoch_buffer, vals)
        # then (possibly) adjusting the xlimits of the plot
        if epoch_buffer[-1] > ax.get_xlim()[1]:
          # increase by the golden ratio
          ax.set_xlim([0, 1.61 * epoch_buffer[-1]])
        # then we look at the y limits which is a bit tricker since
        # we dont' know for sure that the values are monotonic and/or zero origin.
        # first we look at the current limits of the data.
        cur_lim = np.array([np.min(metric_buffer.values),
                            np.max(metric_buffer.values)])
        # then check if any the day falls outside the plot limits
        if cur_lim[0] <= ax.get_ylim()[0] or cur_lim[1] >= ax.get_ylim()[1]:
          # if it does fall outside we increase the y axis by the golden
          # ratio in a way that keeps the data centered
          delta = 1.61 * (np.max(metric_buffer.values) - np.min(metric_buffer.values))
          mid = np.mean(cur_lim)
          new_lim = [mid - delta / 2, mid + delta / 2]
          ax.set_ylim(new_lim)
        # the draw function takes a lot of computation time, by only drawing every
        # so often we can keep the computation bottleneck the filter, not the plotting
        if np.mod(i, draw_every) == 0:
          plt.draw()
          plt.pause(0.001)

  ax.set_xlim([0, epoch_buffer[-1]])
  plt.draw()
  plt.pause(0.001)
  plt.show(block=True)
