template <class T, T... I>
struct Seq {
  static constexpr T PackSize = sizeof...(I);
};

template <typename T, T N>
using MakeSeq = __make_integer_seq<Seq, T, N>;


using SizeT = decltype(sizeof(int));

template <SizeT i, typename ...T>
using TypePackElement = __type_pack_element<i, T...>;

template <int i>
struct X;


template <template <class...> class Templ, class...Types>
using TypeListDedup = __type_list_dedup<Templ, Types...>;

template <class ...Ts>
struct TypeList {};
