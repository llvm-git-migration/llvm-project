%extend lldb::SBAddressRangeList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of address ranges in a lldb.SBAddressRangeList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the address ranges in a lldb.SBAddressRangeList object.'''
      return lldb_iter(self, 'GetSize', 'GetAddressRangeAtIndex')
    %}
#endif
}
