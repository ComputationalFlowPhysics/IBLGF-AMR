#ifndef INCLUDED_CRTP_HPP
#define INCLUDED_CRTP_HPP


namespace crtp
{

template <typename DerivedType, template<typename> class CrtpType>
struct Crtp
{
    DerivedType& derived() { return static_cast<DerivedType&>(*this); }
    DerivedType const& derived() const 
    { 
        return static_cast<DerivedType const&>(*this); 
    }
private:
    Crtp(){}
    friend CrtpType<DerivedType>;
};

/*Example Crtp
template<class T>
struct Base : Crtp<T,Base>
{
    void foo(){ this->derived()->foo(); }
    int a =10;
};
struct Mixin : public Base<Mixin>
{
    void foo() {
        std::cout<<"Mixing using a="
        <<a<<std::endl; 
    }
};
*/

}

#endif
