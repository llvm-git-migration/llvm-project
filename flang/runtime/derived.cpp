//===-- runtime/derived.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "derived.h"
#include "engine.h"
#include "stat.h"
#include "terminator.h"
#include "tools.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

RT_OFFLOAD_API_GROUP_BEGIN

// Fill "extents" array with the extents of component "comp" from derived type
// instance "derivedInstance".
static RT_API_ATTRS void GetComponentExtents(SubscriptValue (&extents)[maxRank],
    const typeInfo::Component &comp, const Descriptor &derivedInstance) {
  const typeInfo::Value *bounds{comp.bounds()};
  for (int dim{0}; dim < comp.rank(); ++dim) {
    auto lb{bounds[2 * dim].GetValue(&derivedInstance).value_or(0)};
    auto ub{bounds[2 * dim + 1].GetValue(&derivedInstance).value_or(0)};
    extents[dim] = ub >= lb ? static_cast<SubscriptValue>(ub - lb + 1) : 0;
  }
}

RT_API_ATTRS auto engine::Initialization::Resume(Engine &engine) -> ResultType {
  while (component_.Iterating(components_)) {
    const auto &comp{
        *componentDesc_->ZeroBasedIndexedElement<typeInfo::Component>(
            component_.at)};
    if (comp.genre() == typeInfo::Component::Genre::Allocatable ||
        comp.genre() == typeInfo::Component::Genre::Automatic) {
      while (element_.Iterating(elements_, &instance_)) {
        Descriptor &allocDesc{*instance_.ElementComponent<Descriptor>(
            element_.subscripts, comp.offset())};
        comp.EstablishDescriptor(allocDesc, instance_, engine.terminator());
        allocDesc.raw().attribute = CFI_attribute_allocatable;
        if (comp.genre() == typeInfo::Component::Genre::Automatic) {
          if (auto stat{ReturnError(engine.terminator(), allocDesc.Allocate(),
                  engine.errMsg(), engine.hasStat())};
              stat != StatOk) {
            return engine.Fail(stat);
          }
          if (const DescriptorAddendum * addendum{allocDesc.Addendum()}) {
            if (const auto *derived{addendum->derivedType()}) {
              if (!derived->noInitializationNeeded()) {
                component_.ResumeAtSameIteration();
                return engine.Begin(Job::Initialization, allocDesc, derived);
              }
            }
          }
        }
      }
    } else if (const void *init{comp.initialization()}) {
      // Explicit initialization of data pointers and
      // non-allocatable non-automatic components
      std::size_t bytes{comp.SizeInBytes(instance_)};
      while (element_.Iterating(elements_, &instance_)) {
        char *ptr{instance_.ElementComponent<char>(
            element_.subscripts, comp.offset())};
        std::memcpy(ptr, init, bytes);
      }
    } else if (comp.genre() == typeInfo::Component::Genre::Pointer) {
      // Data pointers without explicit initialization are established
      // so that they are valid right-hand side targets of pointer
      // assignment statements.
      while (element_.Iterating(elements_, &instance_)) {
        Descriptor &ptrDesc{*instance_.ElementComponent<Descriptor>(
            element_.subscripts, comp.offset())};
        comp.EstablishDescriptor(ptrDesc, instance_, engine.terminator());
        ptrDesc.raw().attribute = CFI_attribute_pointer;
      }
    } else if (comp.genre() == typeInfo::Component::Genre::Data &&
        comp.derivedType() && !comp.derivedType()->noInitializationNeeded()) {
      // Default initialization of non-pointer non-allocatable/automatic
      // data component.  Handles parent component's elements.
      if (!element_.active) {
        GetComponentExtents(extents_, comp, instance_);
      }
      while (element_.Iterating(elements_, &instance_)) {
        Descriptor &compDesc{staticDescriptor_.descriptor()};
        const typeInfo::DerivedType &compType{*comp.derivedType()};
        compDesc.Establish(compType,
            instance_.ElementComponent<char>(
                element_.subscripts, comp.offset()),
            comp.rank(), extents_);
        component_.ResumeAtSameIteration();
        return engine.Begin(Job::Initialization, compDesc, &compType);
      }
    }
  }
  // Initialize procedure pointer components in each element
  const Descriptor &procPtrDesc{derived_->procPtr()};
  std::size_t myProcPtrs{procPtrDesc.Elements()};
  for (std::size_t k{0}; k < myProcPtrs; ++k) {
    const auto &comp{
        *procPtrDesc.ZeroBasedIndexedElement<typeInfo::ProcPtrComponent>(k)};
    while (element_.Iterating(elements_, &instance_)) {
      auto &pptr{*instance_.ElementComponent<typeInfo::ProcedurePointer>(
          element_.subscripts, comp.offset)};
      pptr = comp.procInitialization;
    }
  }
  return engine.Done();
}

RT_API_ATTRS int Initialize(const Descriptor &instance,
    const typeInfo::DerivedType &derived, Terminator &terminator, bool hasStat,
    const Descriptor *errMsg) {
  return engine::Engine{terminator, hasStat, errMsg}.Do(
      engine::Job::Initialization, instance, &derived);
}

static RT_API_ATTRS const typeInfo::SpecialBinding *FindFinal(
    const typeInfo::DerivedType &derived, int rank) {
  if (const auto *ranked{derived.FindSpecialBinding(
          typeInfo::SpecialBinding::RankFinal(rank))}) {
    return ranked;
  } else if (const auto *assumed{derived.FindSpecialBinding(
                 typeInfo::SpecialBinding::Which::AssumedRankFinal)}) {
    return assumed;
  } else {
    return derived.FindSpecialBinding(
        typeInfo::SpecialBinding::Which::ElementalFinal);
  }
}

static RT_API_ATTRS void CallFinalSubroutine(const Descriptor &descriptor,
    const typeInfo::DerivedType &derived, Terminator *terminator) {
  if (const auto *special{FindFinal(derived, descriptor.rank())}) {
    if (special->which() == typeInfo::SpecialBinding::Which::ElementalFinal) {
      std::size_t elements{descriptor.Elements()};
      SubscriptValue at[maxRank];
      descriptor.GetLowerBounds(at);
      if (special->IsArgDescriptor(0)) {
        StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
        Descriptor &elemDesc{statDesc.descriptor()};
        elemDesc = descriptor;
        elemDesc.raw().attribute = CFI_attribute_pointer;
        elemDesc.raw().rank = 0;
        auto *p{special->GetProc<void (*)(const Descriptor &)>()};
        for (std::size_t j{0}; j++ < elements;
             descriptor.IncrementSubscripts(at)) {
          elemDesc.set_base_addr(descriptor.Element<char>(at));
          p(elemDesc);
        }
      } else {
        auto *p{special->GetProc<void (*)(char *)>()};
        for (std::size_t j{0}; j++ < elements;
             descriptor.IncrementSubscripts(at)) {
          p(descriptor.Element<char>(at));
        }
      }
    } else {
      StaticDescriptor<maxRank, true, 10> statDesc;
      Descriptor &copy{statDesc.descriptor()};
      const Descriptor *argDescriptor{&descriptor};
      if (descriptor.rank() > 0 && special->IsArgContiguous(0) &&
          !descriptor.IsContiguous()) {
        // The FINAL subroutine demands a contiguous array argument, but
        // this INTENT(OUT) or intrinsic assignment LHS isn't contiguous.
        // Finalize a shallow copy of the data.
        copy = descriptor;
        copy.set_base_addr(nullptr);
        copy.raw().attribute = CFI_attribute_allocatable;
        Terminator stubTerminator{"CallFinalProcedure() in Fortran runtime", 0};
        RUNTIME_CHECK(terminator ? *terminator : stubTerminator,
            copy.Allocate() == CFI_SUCCESS);
        ShallowCopyDiscontiguousToContiguous(copy, descriptor);
        argDescriptor = &copy;
      }
      if (special->IsArgDescriptor(0)) {
        StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
        Descriptor &tmpDesc{statDesc.descriptor()};
        tmpDesc = *argDescriptor;
        tmpDesc.raw().attribute = CFI_attribute_pointer;
        tmpDesc.Addendum()->set_derivedType(&derived);
        auto *p{special->GetProc<void (*)(const Descriptor &)>()};
        p(tmpDesc);
      } else {
        auto *p{special->GetProc<void (*)(char *)>()};
        p(argDescriptor->OffsetElement<char>());
      }
      if (argDescriptor == &copy) {
        ShallowCopyContiguousToDiscontiguous(descriptor, copy);
        copy.Deallocate();
      }
    }
  }
}

// Fortran 2018 subclause 7.5.6.2
RT_API_ATTRS void Finalize(const Descriptor &descriptor,
    const typeInfo::DerivedType &derived, Terminator *terminator) {
  if (derived.noFinalizationNeeded() || !descriptor.IsAllocated()) {
    return;
  }
  CallFinalSubroutine(descriptor, derived, terminator);
  const auto *parentType{derived.GetParentType()};
  bool recurse{parentType && !parentType->noFinalizationNeeded()};
  // If there's a finalizable parent component, handle it last, as required
  // by the Fortran standard (7.5.6.2), and do so recursively with the same
  // descriptor so that the rank is preserved.
  const Descriptor &componentDesc{derived.component()};
  std::size_t myComponents{componentDesc.Elements()};
  std::size_t elements{descriptor.Elements()};
  for (auto k{recurse ? std::size_t{1}
                      /* skip first component, it's the parent */
                      : 0};
       k < myComponents; ++k) {
    const auto &comp{
        *componentDesc.ZeroBasedIndexedElement<typeInfo::Component>(k)};
    SubscriptValue at[maxRank];
    descriptor.GetLowerBounds(at);
    if (comp.genre() == typeInfo::Component::Genre::Allocatable &&
        comp.category() == TypeCategory::Derived) {
      // Component may be polymorphic or unlimited polymorphic. Need to use the
      // dynamic type to check whether finalization is needed.
      for (std::size_t j{0}; j++ < elements;
           descriptor.IncrementSubscripts(at)) {
        const Descriptor &compDesc{
            *descriptor.ElementComponent<Descriptor>(at, comp.offset())};
        if (compDesc.IsAllocated()) {
          if (const DescriptorAddendum * addendum{compDesc.Addendum()}) {
            if (const typeInfo::DerivedType *
                compDynamicType{addendum->derivedType()}) {
              if (!compDynamicType->noFinalizationNeeded()) {
                Finalize(compDesc, *compDynamicType, terminator);
              }
            }
          }
        }
      }
    } else if (comp.genre() == typeInfo::Component::Genre::Allocatable ||
        comp.genre() == typeInfo::Component::Genre::Automatic) {
      if (const typeInfo::DerivedType * compType{comp.derivedType()}) {
        if (!compType->noFinalizationNeeded()) {
          for (std::size_t j{0}; j++ < elements;
               descriptor.IncrementSubscripts(at)) {
            const Descriptor &compDesc{
                *descriptor.ElementComponent<Descriptor>(at, comp.offset())};
            if (compDesc.IsAllocated()) {
              Finalize(compDesc, *compType, terminator);
            }
          }
        }
      }
    } else if (comp.genre() == typeInfo::Component::Genre::Data &&
        comp.derivedType() && !comp.derivedType()->noFinalizationNeeded()) {
      SubscriptValue extents[maxRank];
      GetComponentExtents(extents, comp, descriptor);
      StaticDescriptor<maxRank, true, 0> staticDescriptor;
      Descriptor &compDesc{staticDescriptor.descriptor()};
      const typeInfo::DerivedType &compType{*comp.derivedType()};
      for (std::size_t j{0}; j++ < elements;
           descriptor.IncrementSubscripts(at)) {
        compDesc.Establish(compType,
            descriptor.ElementComponent<char>(at, comp.offset()), comp.rank(),
            extents);
        Finalize(compDesc, compType, terminator);
      }
    }
  }
  if (recurse) {
    StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
    Descriptor &tmpDesc{statDesc.descriptor()};
    tmpDesc = descriptor;
    tmpDesc.raw().attribute = CFI_attribute_pointer;
    tmpDesc.Addendum()->set_derivedType(parentType);
    tmpDesc.raw().elem_len = parentType->sizeInBytes();
    Finalize(tmpDesc, *parentType, terminator);
  }
}

// The order of finalization follows Fortran 2018 7.5.6.2, with
// elementwise finalization of non-parent components taking place
// before parent component finalization, and with all finalization
// preceding any deallocation.
RT_API_ATTRS void Destroy(const Descriptor &descriptor, bool finalize,
    const typeInfo::DerivedType &derived, Terminator *terminator) {
  if (derived.noDestructionNeeded() || !descriptor.IsAllocated()) {
    return;
  }
  if (finalize && !derived.noFinalizationNeeded()) {
    Finalize(descriptor, derived, terminator);
  }
  // Deallocate all direct and indirect allocatable and automatic components.
  // Contrary to finalization, the order of deallocation does not matter.
  const Descriptor &componentDesc{derived.component()};
  std::size_t myComponents{componentDesc.Elements()};
  std::size_t elements{descriptor.Elements()};
  SubscriptValue at[maxRank];
  descriptor.GetLowerBounds(at);
  for (std::size_t k{0}; k < myComponents; ++k) {
    const auto &comp{
        *componentDesc.ZeroBasedIndexedElement<typeInfo::Component>(k)};
    const bool destroyComp{
        comp.derivedType() && !comp.derivedType()->noDestructionNeeded()};
    if (comp.genre() == typeInfo::Component::Genre::Allocatable ||
        comp.genre() == typeInfo::Component::Genre::Automatic) {
      for (std::size_t j{0}; j < elements; ++j) {
        Descriptor *d{
            descriptor.ElementComponent<Descriptor>(at, comp.offset())};
        if (destroyComp) {
          Destroy(*d, /*finalize=*/false, *comp.derivedType(), terminator);
        }
        d->Deallocate();
        descriptor.IncrementSubscripts(at);
      }
    } else if (destroyComp &&
        comp.genre() == typeInfo::Component::Genre::Data) {
      SubscriptValue extents[maxRank];
      GetComponentExtents(extents, comp, descriptor);
      StaticDescriptor<maxRank, true, 0> staticDescriptor;
      Descriptor &compDesc{staticDescriptor.descriptor()};
      const typeInfo::DerivedType &compType{*comp.derivedType()};
      for (std::size_t j{0}; j++ < elements;
           descriptor.IncrementSubscripts(at)) {
        compDesc.Establish(compType,
            descriptor.ElementComponent<char>(at, comp.offset()), comp.rank(),
            extents);
        Destroy(compDesc, /*finalize=*/false, *comp.derivedType(), terminator);
      }
    }
  }
}

RT_API_ATTRS bool HasDynamicComponent(const Descriptor &descriptor) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived = addendum->derivedType()) {
      // Destruction is needed if and only if there are direct or indirect
      // allocatable or automatic components.
      return !derived->noDestructionNeeded();
    }
  }
  return false;
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
