* Added an opt-in single-world evaluation reset schedule that isolated each
  episode's Torch randomization from policy sampling and rollout length.
  Physical start hashes were checked before scenario metadata was written;
  terminal automatic resets were recorded without counting extra episodes.
  Legacy evaluation behavior and training checkpoints were preserved by default.
