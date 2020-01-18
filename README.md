# harmonic-enhancer
Shure SM58 frequency response replica for your vocals using Python 3.

Edit the the following line in harmonic_enhancer.py with the name of your WAV file, make sure that the enhancer file is in the same directory as your sound file.
Fs, x_t = wavfile.read("YourFileName.wav")

Time domain & Frequency domain graphs of your original and improved audio files will be plotted along with an audio spectrogram and the frequency response of the enhancer.

The improved audio file will be saved as improved.WAV into the same local directory.
