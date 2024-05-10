%extend lldb::SBAddressRangeList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of address ranges in a lldb.SBAddressRangeList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the address ranges in a lldb.SBAddressRangeList object.'''
      return lldb_iter(self, 'GetSize', 'GetAddressRangeAtIndex')

    def __getitem__(self, idx):
      '''Get the address range at a given index in an lldb.SBAddressRangeList object.'''
      if type(idx) == int:
        if idx >= self.GetSize():
          raise IndexError("list index out of range")
        return self.GetAddressRangeAtIndex(idx)
      else:
        print("error: unsupported idx type: %s" % type(key))
        return None
    def __repr__(self):
      import lldb
      stream = lldb.SBStream()
      self.GetDescription(stream, lldb.target)
      return stream.GetData()
    %}
#endif
}
