import pickle


# The advantage of HIGHEST_PROTOCOL is that files get smaller. This makes unpickling sometimes much faster.

# Important notice: The maximum file size of pickle is about 2GB.

# Alternatives
# CSV: Super simple format (read & write)
# JSON: Nice for writing human-readable data; VERY commonly used (read & write)
# YAML: YAML is a superset of JSON, but easier to read (read & write, comparison of JSON and YAML)
# pickle: A Python serialization format (read & write)
# MessagePack (Python package): More compact representation (read & write)
# HDF5 (Python package): Nice for matrices (read & write)
# XML: exists too *sigh* (read & write)
# For your application, the following might be important:

# Support by other programming languages
# Reading / writing performance
# Compactness (file size)


your_data = {'foo': 'bar'}

# Store data (serialize)
with open('filename.pickle', 'wb') as handle:
    pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
with open('filename.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)

print(your_data == unserialized_data)

