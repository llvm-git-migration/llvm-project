
#ifndef LLVM_CLANG_SEMA_ATTR_TEMPLATE_INSTANTIATE_H
#define LLVM_CLANG_SEMA_ATTR_TEMPLATE_INSTANTIATE_H

#include "clang/AST/Attr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"

// from AttrTemplateInstantiate.inc
namespace clang::sema {
Attr *instantiateTemplateAttribute(
    const Attr *At, ASTContext &C, Sema &S,
    const MultiLevelTemplateArgumentList &TemplateArgs);
Attr *instantiateTemplateAttributeForDecl(
    const Attr *At, ASTContext &C, Sema &S,
    const MultiLevelTemplateArgumentList &TemplateArgs);
} // namespace clang::sema

#endif // LLVM_CLANG_SEMA_ATTR_TEMPLATE_INSTANTIATE_H
