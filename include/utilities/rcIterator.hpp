#ifndef INCLUDED_LGF_RCITERATOR_HPP
#define INCLUDED_LGF_RCITERATOR_HPP


template<int Dim, int D=Dim-1>
struct rcIterator
{

    template<class ArrayType, class Function>
    static void iterate(const ArrayType& _base,
                      const ArrayType& _extent,
                      const Function& f)
    {
        auto p=_base;
        rcIterator<Dim, D>::apply_impl(p,f,_base, _extent, ArrayType(1));
    }
    template<class ArrayType, class Function>
    static void iterate(const ArrayType& _base,
                      const ArrayType& _extent,
                      const ArrayType& _stride, 
                      const Function& f)
    {
        auto p=_base;
        rcIterator<Dim, D>::apply_impl(p,f,_base, _extent, _stride );
    }

    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent,
                           const ArrayType& _stride
                       )
    {
        for(std::size_t k=0; k<static_cast<std::size_t>(_extent[D]);k+=_stride[D])
        {
            _p[D]=_base[D]+k;
            rcIterator<Dim, D-1>::apply_impl(_p,f, _base, _extent, _stride);
        }
    }
};

template<int Dim>
struct rcIterator<Dim,0>
{
    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent,
                           const ArrayType& _stride
                           )
    {
        for(std::size_t k=0; k<static_cast<std::size_t>(_extent[0]);k+=_stride[0])
        {
            _p[0]=_base[0]+k;
            f(_p);
        }
    }
};


#endif 
