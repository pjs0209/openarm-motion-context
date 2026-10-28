* Skipped Kit callback pumping and graceful GPU cleanup after a fatal CUDA
  exception so a poisoned RTX context could not leave failed simulations
  hanging during shutdown. Preserved the original failure exit status.
