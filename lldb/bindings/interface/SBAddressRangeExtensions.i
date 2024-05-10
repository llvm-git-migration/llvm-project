%extend lldb::SBAddressRange {
#ifdef SWIGPYTHON
    %pythoncode%{
      def __repr__(self):
        import lldb
        stream = lldb.SBStream()
        self.GetDescription(stream, lldb.target)
        return stream.GetData()
    %}
#endif
}
