//! A syntax tree to ops.json, checked against the project's types as it goes.
//!
//! Every statement becomes nodes on the exec chain, and every expression a tree of nodes feeding
//! it. A pure call is a node off the chain, evaluated when read. An impure call inside an
//! expression is put on the chain when the expression is lowered, which is before the statement
//! that uses it, so its result is computed once and in source order. Literals become pin defaults
//! rather than nodes. Event bodies are lowered into functions of their own, which the event node
//! calls, so all code has locals and one shape. The exception is a handler that waits, which only
//! the event graph can do: it is lowered there, and its locals become variables of the Blueprint.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};

use crate::builtins::{self, Lowering, On, ARRAY, MAP, MATH, SYSTEM};
use crate::ops::{ComponentField, Document, Enumerator, Local, StructField, NodeKind, Op, Param, PinRef, Scope, VERSION};
use crate::syntax::{self, Access, BinaryOp, Enum, Expr, Interface, Purity, Struct, ExprKind, File, Item, ParamDecl, Span, Stmt, TypeName, UnaryOp, RESERVED_PREFIX};
use crate::types::{delegate_name, Declared, Event, Floats, Function, Passing, Slot, Ty, TypeDb, TypeScope, INTERFACE};

const OBJECT: &str = "/Script/CoreUObject.Object";
const EVENT_GRAPH: &str = "EventGraph";
const KEY: &str = "/Script/InputCore.Key";
const LINEAR_COLOR: &str = "/Script/CoreUObject.LinearColor";
const SLATE_COLOR: &str = "/Script/SlateCore.SlateColor";
/// The name of the output pin that carries a loom function's return value.
const RETURN_PIN: &str = "ReturnValue";

/// What feeds an input pin: another node's output, or a literal the pin holds as its default.
#[derive(Debug, Clone)]
enum Source {
    Pin(PinRef),
    Literal(String),
    /// Nothing: the pin keeps its own default, which for an object is none.
    Nothing,
    /// A node whose several outputs are read by name, as a record.
    Node(String),
}

#[derive(Debug, Clone)]
struct Value {
    source: Source,
    ty: Ty,
}

struct UserFunction {
    /// Each input's name, type, and whether it is passed by reference.
    params: Vec<(String, Ty, bool)>,
    /// A pure node, with no exec pins.
    pure: bool,
    /// Each input's default as text, which a call that leaves it out gives it.
    defaults: Vec<Option<String>>,
    ret: Option<Ty>,
    /// Its `out` parameters, in the order declared, which a call gives as a record with the return
    /// value, as an engine function's outputs are given.
    outputs: Vec<(String, Ty)>,
}

struct Program<'a> {
    db: &'a TypeDb,
    /// An editor Blueprint, which may use the classes and functions that run only in the editor.
    editor: bool,
    /// The Blueprint's own class, which `self` has; none in a function library, which has no self.
    class: Option<String>,
    /// The class that calls a function, for whether it may: the Blueprint's own, which a function
    /// library has too.
    caller: Option<String>,
    parent: String,
    members: HashMap<String, Ty>,
    functions: HashMap<String, UserFunction>,
    /// Each event dispatcher's parameters.
    dispatchers: HashMap<String, Vec<(String, Ty)>>,
    /// The folder of the Blueprint's package, where a type named by its short name is looked for
    /// first.
    folder: String,
    /// The folders the source uses, where a type is looked for next.
    uses: Vec<String>,
    /// The interfaces the Blueprint implements: those it names with `implements`, then its
    /// parent's.
    interfaces: Vec<String>,
    /// Each of the Blueprint's functions that is bound to a delegate, with how wide the floats
    /// it takes are, as the first delegate it is bound to passes them.
    narrowed: RefCell<HashMap<String, Narrowed>>,
}

/// How wide the floats a function bound to a delegate takes are, and where it was first bound.
struct Narrowed {
    /// For each of its parameters and its return value, in order.
    widths: Vec<Floats>,
    span: Span,
    /// What it was bound to, as a message names it.
    to: String,
}

const USER_WIDGET: &str = "/Script/UMG.UserWidget";
const ACTOR: &str = "/Script/Engine.Actor";
const ACTOR_COMPONENT: &str = "/Script/Engine.ActorComponent";
/// What holds a widget placed on the root canvas.
const CANVAS_SLOT: &str = "/Script/UMG.CanvasPanelSlot";
const SUBSYSTEMS: [&str; 2] = ["/Script/Engine.EngineSubsystem", "/Script/EditorSubsystem.EditorSubsystem"];
const FUNCTION_LIBRARY: &str = "/Script/Engine.BlueprintFunctionLibrary";

/// Why a component or widget of a class is refused: the project allows only Blueprint classes under
/// its InstancedClassPaths, such as when some of its Blueprint classes only stand in for classes
/// defined elsewhere, with a property layout of their own.
fn not_instanced(what: &str, class: &str, db: &TypeDb) -> String {
    format!(
        "{} is a Blueprint class outside the project's InstancedClassPaths ({}); a {what} must be a native class or a Blueprint class under one of them, as set in Project Settings > Plugins > Loom",
        short(class),
        db.instanced_class_paths().join(", ")
    )
}

/// The class a Blueprint's source declares, as other code names it.
pub fn class_path(file: &File) -> Result<String> {
    match file.path.rsplit_once('/') {
        Some((_, asset)) if asset == file.name => Ok(format!("{}.{}_C", file.path, file.name)),
        _ => bail!("{} must be saved as .../{}, not {}", file.name, file.name, file.path),
    }
}

/// How the types and ops.json name a function's access, when it isn't public.
fn access_word(access: Access) -> Option<&'static str> {
    match access {
        Access::Public => None,
        Access::Protected => Some("protected"),
        Access::Private => Some("private"),
    }
}

/// The enum an enum's source declares, as other code names it.
pub fn enum_path(e: &Enum) -> Result<String> {
    match e.path.rsplit_once('/') {
        Some((_, asset)) if asset == e.name => Ok(format!("{}.{}", e.path, e.name)),
        _ => bail!("{} must be saved as .../{}, not {}", e.name, e.name, e.path),
    }
}

/// The struct a struct's source declares, as other code names it.
pub fn struct_path(s: &Struct) -> Result<String> {
    match s.path.rsplit_once('/') {
        Some((_, asset)) if asset == s.name => Ok(format!("{}.{}", s.path, s.name)),
        _ => bail!("{} must be saved as .../{}, not {}", s.name, s.name, s.path),
    }
}

/// The class an interface's source declares, as other code names it.
pub fn interface_path(i: &Interface) -> Result<String> {
    match i.path.rsplit_once('/') {
        Some((_, asset)) if asset == i.name => Ok(format!("{}.{}_C", i.path, i.name)),
        _ => bail!("{} must be saved as .../{}, not {}", i.name, i.name, i.path),
    }
}

/// The asset a source builds, as other code names it: a Blueprint's or an interface's class, an
/// enum or a struct.
pub fn asset_path(source: &syntax::Source) -> Result<String> {
    match source {
        syntax::Source::Blueprint(file) => class_path(file),
        syntax::Source::Enum(e) => enum_path(e),
        syntax::Source::Struct(s) => struct_path(s),
        syntax::Source::Interface(i) => interface_path(i),
    }
}

/// An interface's functions, as the Blueprints that implement it see them.
fn interface_functions(i: &Interface, program: &Program) -> Result<Vec<Declared>> {
    let mut names = HashSet::new();
    let mut functions = Vec::new();
    for function in &i.functions {
        if !names.insert(&function.name) {
            bail!("{}: {} is declared twice", function.span, function.name);
        }
        if let Some(param) = function.params.iter().find(|p| p.by_ref) {
            bail!("{}: {}'s parameter {} is a ref, which an interface's function cannot take yet", function.span, function.name, param.name);
        }
        let mut seen = HashSet::new();
        if let Some(param) = function.params.iter().find(|p| !seen.insert(p.name.clone())) {
            bail!("{}: {} has two parameters called {}", function.span, function.name, param.name);
        }
        let params = program.params(&function.params)?;
        let outputs = program.outputs(&function.params)?;
        let ret = function.ret.as_ref().map(|t| program.ty(t)).transpose()?;
        let defaults = vec![None; params.len()];
        functions.push(Declared { name: function.name.clone(), params, defaults, ret, outputs, is_static: false, pure: false, access: None });
    }
    Ok(functions)
}

/// What finds the types a struct's source names: its own folder first, like a Blueprint's.
fn type_source_program<'a>(db: &'a TypeDb, path: &str) -> Program<'a> {
    Program {
        db,
        editor: false,
        class: None,
        caller: None,
        parent: String::new(),
        members: HashMap::new(),
        functions: HashMap::new(),
        dispatchers: HashMap::new(),
        folder: path.rsplit_once('/').map_or(String::new(), |(folder, _)| folder.to_string()),
        uses: Vec::new(),
        interfaces: Vec::new(),
        narrowed: RefCell::default(),
    }
}

/// The folder of a source's package, such as /Game/Things for /Game/Things/BP_Thing.
fn folder_of(file: &File) -> String {
    file.path.rsplit_once('/').map_or(String::new(), |(folder, _)| folder.to_string())
}

/// The class a Blueprint's source names as its parent. An editor Blueprint, and only one, has a
/// parent that runs only in the editor.
fn parent_of(file: &File, db: &TypeDb) -> Result<String> {
    // Looked up with the editor's classes too, so the check below can say which way it is wrong.
    let parent = db
        .class_in(&file.parent, true, &TypeScope { folder: &folder_of(file), uses: &file.uses() })
        .map_err(|e| anyhow!("{}: {e}", file.name))?
        .ok_or_else(|| anyhow!("{}: the parent class {} is not in the types", file.name, file.parent))?;
    match (file.editor, db.is_editor_only(&parent)) {
        (true, false) => bail!("{}: {} runs in the game, so this is not an editor Blueprint; remove `editor`", file.name, file.parent),
        (false, true) => bail!("{}: {} runs only in the editor; write `editor blueprint` for a Blueprint that does too", file.name, file.parent),
        _ => Ok(parent),
    }
}

/// Adds the Blueprints, enums and structs of a compile to the types, so each can use the others:
/// their variables, widgets and functions, their values and their fields.
pub fn declare(sources: &[syntax::Source], db: &mut TypeDb) -> Result<(), DeclareError> {
    // The enums first, since they use nothing, then the names of the structs, interfaces and
    // Blueprints, so any of them can have the type of another, then the structs' fields, the
    // interfaces' functions and the Blueprints' members.
    for (index, source) in sources.iter().enumerate() {
        let err = |error| DeclareError { file: index, error };
        match source {
            syntax::Source::Enum(e) => db.add_enum(&enum_path(e).map_err(err)?, e.values.iter().map(|v| v.name.clone()).collect()),
            syntax::Source::Struct(s) => db.add_struct(&struct_path(s).map_err(err)?, &[]),
            syntax::Source::Interface(i) => db.add_interface(&interface_path(i).map_err(err)?, &[]),
            syntax::Source::Blueprint(_) => {}
        }
    }
    let files: Vec<(usize, &File)> = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| match source {
            syntax::Source::Blueprint(file) => Some((index, file)),
            _ => None,
        })
        .collect();
    // Names first, so a Blueprint's members can have the type of one declared after it. A
    // Blueprint whose parent is another source's is declared after that one, whatever their order,
    // so each pass declares those whose parents are known, until one declares nothing.
    let mut classes = Vec::new();
    for &(index, file) in &files {
        classes.push(class_path(file).map_err(|error| DeclareError { file: index, error })?);
    }
    let mut waiting: Vec<usize> = (0..files.len()).collect();
    while !waiting.is_empty() {
        let before = waiting.len();
        waiting.retain(|&index| match parent_of(files[index].1, db) {
            Ok(parent) => {
                db.add_class(&classes[index], &parent, &[], &[], &[], &[]);
                false
            }
            Err(_) => true,
        });
        if waiting.len() == before {
            // Nothing more can be declared: the first left has no parent, or is in a cycle.
            let (index, file) = files[waiting[0]];
            let in_cycle = waiting.iter().any(|&other| {
                files[other].1.name == file.parent || classes[other].rsplit_once('.').is_some_and(|(package, _)| package == file.parent)
            });
            let error = match parent_of(file, db) {
                Err(_) if in_cycle => anyhow!("{}: its parent {} comes back round to it through the sources' parents", file.name, file.parent),
                Err(error) => error,
                Ok(_) => unreachable!("a parent that was not known is known without a declaration"),
            };
            return Err(DeclareError { file: index, error });
        }
    }
    for (index, source) in sources.iter().enumerate() {
        let err = |error| DeclareError { file: index, error };
        match source {
            syntax::Source::Struct(s) => {
                let program = type_source_program(db, &s.path);
                let fields: Vec<(String, Ty)> = s.fields.iter().map(|f| Ok((f.name.clone(), program.ty(&f.ty)?))).collect::<Result<_>>().map_err(err)?;
                db.add_struct(&struct_path(s).map_err(err)?, &fields);
            }
            syntax::Source::Interface(i) => {
                let functions = interface_functions(i, &type_source_program(db, &i.path)).map_err(err)?;
                db.add_interface(&interface_path(i).map_err(err)?, &functions);
            }
            _ => {}
        }
    }
    for &(index, file) in &files {
        declare_members(file, db).map_err(|error| DeclareError { file: index, error })?;
    }
    Ok(())
}

/// Why a compile's sources could not be declared to each other, and which of them it was.
#[derive(Debug)]
pub struct DeclareError {
    /// The source's place in the list declared.
    pub file: usize,
    pub error: anyhow::Error,
}

impl std::fmt::Display for DeclareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.error)
    }
}

impl std::error::Error for DeclareError {}

/// Declares one source's members, once every source's class is declared.
fn declare_members(file: &File, db: &mut TypeDb) -> Result<()> {
    let parent = parent_of(file, db)?;
    let program = Program { db, editor: file.editor, class: None, caller: None, parent: parent.clone(), members: HashMap::new(), functions: HashMap::new(), dispatchers: HashMap::new(), folder: folder_of(file), uses: file.uses(), interfaces: Vec::new(), narrowed: RefCell::default() };
    let mut properties = Vec::new();
    let mut functions = Vec::new();
    let mut dispatchers = Vec::new();
    let mut interfaces = Vec::new();
    for item in &file.items {
        match item {
            Item::Implements { interface, span } => {
                let path = program.implemented(interface, *span, &interfaces)?;
                interfaces.push(path);
            }
            Item::Var { name, ty, private, .. } => properties.push((name.clone(), program.ty(ty)?, *private)),
            Item::Widget { name, class, .. } | Item::Component { name, class, .. } => {
                properties.push((name.clone(), program.ty(class)?, false))
            }
            Item::Dispatcher { name, params, .. } => {
                let params = program.params(params)?.into_iter().map(|(n, t, _)| (n, t)).collect();
                dispatchers.push((name.clone(), params));
            }
            Item::Event { name, params, .. } => functions.push(Declared {
                name: name.clone(),
                pure: false,
                access: None,
                defaults: vec![None; params.len()],
                params: program.params(params)?,
                ret: None,
                outputs: Vec::new(),
                is_static: false,
            }),
            Item::Fn { name, params, ret, access, pure, .. } => functions.push(Declared {
                name: name.clone(),
                pure: *pure != Purity::Impure,
                access: access_word(*access),
                defaults: program.defaults(params)?,
                params: program.params(params)?,
                ret: ret.as_ref().map(|t| program.ty(t)).transpose()?,
                outputs: program.outputs(params)?,
                is_static: db.is_subclass(&parent, FUNCTION_LIBRARY),
            }),
            Item::On { .. } | Item::Category { .. } | Item::Default { .. } | Item::Use { .. } => {}
        }
    }
    db.add_class(&class_path(file)?, &parent, &properties, &functions, &dispatchers, &interfaces);
    Ok(())
}

/// A source to the document that builds its Blueprint or enum.
pub fn lower_source(source: &syntax::Source, file: &str, db: &TypeDb) -> Result<Document> {
    match source {
        syntax::Source::Blueprint(blueprint) => lower(blueprint, file, db),
        syntax::Source::Enum(e) => lower_enum(e, file),
        syntax::Source::Struct(s) => lower_struct(s, file, db),
        syntax::Source::Interface(i) => lower_interface(i, file, db),
    }
}

/// An interface's source to the document that builds its Blueprint interface: a function with no
/// body for each it declares.
pub fn lower_interface(i: &Interface, source: &str, db: &TypeDb) -> Result<Document> {
    if i.functions.is_empty() {
        bail!("{}: {} has no functions; an interface declares at least one", i.span, i.name);
    }
    let functions = interface_functions(i, &type_source_program(db, &i.path))?;
    let mut ops = vec![Op::Blueprint { path: i.path.clone(), parent: INTERFACE.to_string(), source: Some(source.to_string()), widget: false, canvas: None }];
    for (function, signature) in functions.iter().zip(&i.functions) {
        ops.push(Op::Function {
            name: function.name.clone(),
            params: function
                .params
                .iter()
                .map(|(n, t, by_ref)| Ok(Param { name: n.clone(), ty: t.op_type()?, by_ref: *by_ref, default: None }))
                .collect::<Result<_>>()?,
            // The return value's pin first, then a pin per `out` parameter, as a function's are.
            returns: function
                .ret
                .iter()
                .map(|t| (RETURN_PIN, t))
                .chain(function.outputs.iter().map(|(n, t)| (n.as_str(), t)))
                .map(|(n, t)| Ok(Param { name: n.to_string(), ty: t.op_type()?, by_ref: false, default: None }))
                .collect::<Result<_>>()?,
            locals: Vec::new(),
            description: signature.doc.clone(),
            category: None,
            overrides: None,
            pure: false,
            access: None,
        });
    }
    ops.push(Op::Compile);
    ops.push(Op::Save);
    Ok(Document { version: VERSION, ops })
}

/// A struct's source to the document that makes it.
pub fn lower_struct(s: &Struct, source: &str, db: &TypeDb) -> Result<Document> {
    if s.fields.is_empty() {
        bail!("{}: {} has no fields; a struct has at least one", s.span, s.name);
    }
    let own = struct_path(s)?;
    let program = type_source_program(db, &s.path);
    let mut names = HashSet::new();
    let mut fields = Vec::new();
    for field in &s.fields {
        if !names.insert(&field.name) {
            bail!("{}: {} is declared twice", field.span, field.name);
        }
        let ty = program.ty(&field.ty)?;
        if matches!(&ty, Ty::Struct(path) if *path == own) {
            bail!("{}: {} is a {}, which would hold itself; a struct cannot", field.span, field.name, s.name);
        }
        if let Ty::Delegate(_) = ty {
            bail!("{}: {} is a {ty}, and the struct editor refuses a delegate field", field.span, field.name);
        }
        let default = field.default.as_ref().map(|e| literal(e, &ty, &program)).transpose()?;
        fields.push(StructField { name: field.name.clone(), ty: ty.op_type()?, default, description: field.doc.clone() });
    }
    Ok(Document {
        version: VERSION,
        ops: vec![Op::Struct { path: s.path.clone(), source: Some(source.to_string()), fields, description: s.doc.clone() }, Op::Save],
    })
}

/// An enum's source to the document that makes it. It holds no more than 255 values, since a
/// Blueprint keeps an enum's value in a byte and the last is its generated `_MAX`.
pub fn lower_enum(e: &Enum, source: &str) -> Result<Document> {
    if e.values.is_empty() {
        bail!("{}: {} has no values; an enum has at least one", e.span, e.name);
    }
    if e.values.len() > 255 {
        bail!("{}: {} has {} values, and an enum has at most 255", e.span, e.name, e.values.len());
    }
    let mut names = HashSet::new();
    let mut shown: HashMap<&str, &str> = HashMap::new();
    for value in &e.values {
        if !names.insert(&value.name) {
            bail!("{}: {} is listed twice", value.span, value.name);
        }
        let label = value.label.as_deref().unwrap_or(&value.name);
        if label.trim().is_empty() {
            bail!("{}: {}'s label is empty", value.span, value.name);
        }
        if let Some(other) = shown.insert(label, &value.name) {
            bail!("{}: {} and {other} would both be shown as \"{label}\"; give each a label of its own", value.span, value.name);
        }
    }
    let values = e
        .values
        .iter()
        .map(|v| Enumerator { name: v.name.clone(), label: v.label.clone(), description: v.doc.clone() })
        .collect();
    Ok(Document {
        version: VERSION,
        ops: vec![Op::Enum { path: e.path.clone(), source: Some(source.to_string()), values, description: e.doc.clone() }, Op::Save],
    })
}

pub fn lower(file: &File, source: &str, db: &TypeDb) -> Result<Document> {
    let parent = parent_of(file, db)?;
    let widget = db.is_subclass(&parent, USER_WIDGET);
    // A function library holds only functions, which are static and have no object to work on.
    let library = db.is_subclass(&parent, FUNCTION_LIBRARY);
    // The editor makes a library only from a C++ library with no functions of its own, such as
    // BlueprintFunctionLibrary or EditorFunctionLibrary. It won't make a child of another library,
    // Blueprint or C++, so neither does loom.
    if library && (!parent.starts_with("/Script/") || db.declares_functions(&parent)) {
        bail!("{}: a function library can't derive from another library, and {} is one; its parent must be BlueprintFunctionLibrary", file.name, file.parent);
    }
    // `self` has the Blueprint's own class when this compile declared it, and its parent's otherwise.
    let own = class_path(file)?;
    let caller = if db.class_in(&own, file.editor, &TypeScope { folder: &folder_of(file), uses: &file.uses() }).ok().flatten().is_some() { own } else { parent.clone() };
    let class = (!library).then(|| caller.clone());
    let mut program = Program { db, editor: file.editor, class, caller: Some(caller), parent: parent.clone(), members: HashMap::new(), functions: HashMap::new(), dispatchers: HashMap::new(), folder: folder_of(file), uses: file.uses(), interfaces: Vec::new(), narrowed: RefCell::default() };
    let canvas = widget.then(|| format!("{RESERVED_PREFIX}Canvas"));
    let mut ops = vec![Op::Blueprint { path: file.path.clone(), parent, source: Some(source.to_string()), widget, canvas }];

    // The interfaces first, since no member may take the name of one of their functions.
    for item in &file.items {
        if let Item::Implements { interface, span } = item {
            let path = program.implemented(interface, *span, &program.interfaces)?;
            ops.push(Op::Implement { interface: path.clone() });
            program.interfaces.push(path);
        }
    }
    program.interfaces.extend(db.interfaces_of(&program.parent));

    let mut category = None;
    for item in &file.items {
        if let Item::Category { name, span } = item {
            if category.replace(name.clone()).is_some() {
                bail!("{span}: a Blueprint has one category");
            }
        }
    }

    // Components first, so a handler can name one declared after it.
    for item in &file.items {
        if let Item::Component { name, class, fields, span } = item {
            if !db.is_subclass(&program.parent, ACTOR) {
                bail!("{span}: only an actor's Blueprint has components");
            }
            let ty = program.ty(class)?;
            let Ty::Object(class) = ty.clone() else { bail!("{span}: {} is not a component class", class.name) };
            if !db.is_subclass(&class, ACTOR_COMPONENT) {
                bail!("{span}: {} is not a component", short(&class));
            }
            if !db.may_instance(&class) {
                bail!("{span}: {}", not_instanced("component", &class, db));
            }
            if program.members.contains_key(name) {
                bail!("{span}: {name} is declared twice");
            }
            program.inherited(name, *span)?;
            let mut defaults = Vec::new();
            for (field, value) in fields {
                let Some(property) = db.editable_property(&class, field) else {
                    bail!("{}: {} has no {field}", value.span, short(&class));
                };
                defaults.push(ComponentField { name: field.clone(), value: literal(value, &property.ty, &program)? });
            }
            ops.push(Op::Component { name: name.clone(), class: class.clone(), fields: defaults });
            program.members.insert(name.clone(), ty);
        }
    }

    // Declarations first, so any body can use any variable or function.
    let mut overrides = HashMap::new();
    // Each widget's class by name, for its children to find the slot it gives them.
    let mut widget_classes: HashMap<String, String> = HashMap::new();
    for item in &file.items {
        match item {
            Item::Category { .. } | Item::Component { .. } | Item::Default { .. } | Item::Implements { .. } => {}
            // A used folder with nothing in it is a mistake, such as a version that isn't there.
            // It's noted as missing, so whatever loads the types can load what's under it.
            Item::Use { folder, span } => {
                if !db.holds_anything_under(folder) {
                    db.note_missing_folder(folder);
                    bail!("{span}: nothing a source can use is under {folder}: no Blueprint, struct or enum");
                }
            }
            Item::Var { name, ty, default, editable, private, doc, span } => {
                if library {
                    bail!("{span}: a function library has no variables");
                }
                let ty = program.ty(ty)?;
                if let Ty::Delegate(_) = ty {
                    bail!("{span}: {name} cannot be a {ty}, since Blueprint has no delegate variables; take one as a function's parameter and pass it on");
                }
                if program.members.contains_key(name) || program.functions.contains_key(name) {
                    bail!("{span}: {name} is declared twice");
                }
                program.inherited(name, *span)?;
                let default = default.as_ref().map(|e| literal(e, &ty, &program)).transpose()?;
                ops.push(Op::Variable {
                    name: name.clone(),
                    ty: ty.op_type()?,
                    default,
                    editable: *editable,
                    private: *private,
                    description: doc.clone(),
                    category: category.clone(),
                });
                program.members.insert(name.clone(), ty);
            }
            Item::Dispatcher { name, params, doc, span } => {
                if library {
                    bail!("{span}: a function library has no dispatchers");
                }
                if program.members.contains_key(name) || program.functions.contains_key(name) || program.dispatchers.contains_key(name) {
                    bail!("{span}: {name} is declared twice");
                }
                program.inherited(name, *span)?;
                let params: Vec<(String, Ty)> = program.params(params)?.into_iter().map(|(n, t, _)| (n, t)).collect();
                ops.push(Op::Dispatcher {
                    name: name.clone(),
                    params: params
                        .iter()
                        .map(|(n, t)| Ok(Param { name: n.clone(), ty: t.op_type()?, by_ref: false, default: None }))
                        .collect::<Result<_>>()?,
                    description: doc.clone(),
                    category: category.clone(),
                });
                program.dispatchers.insert(name.clone(), params);
            }
            Item::Widget { name, class, fields, slot, place, parent, span } => {
                if !widget {
                    bail!("{span}: only a widget Blueprint, whose parent is a UserWidget, has widgets");
                }
                let ty = program.ty(class)?;
                let Ty::Object(class) = &ty else { bail!("{span}: {} is not a widget class", class.name) };
                if !db.may_instance(class) {
                    bail!("{span}: {}", not_instanced("widget", class, db));
                }
                if program.members.contains_key(name) {
                    bail!("{span}: {name} is declared twice");
                }
                program.inherited(name, *span)?;
                // What holds the widget: its parent's slot, the root canvas's, or nothing for the root.
                let slot_class = match (parent, place) {
                    (Some(parent), Some(_)) => {
                        bail!("{span}: {name} is inside {parent}, so it has no place on the root canvas; its slot places it")
                    }
                    (Some(parent), None) => {
                        let parent_class = widget_classes[parent].clone();
                        let Some((slot_class, one_child)) = db.panel(&parent_class) else {
                            bail!("{span}: {parent} is a {}, which holds no widgets", short(&parent_class));
                        };
                        let siblings = file.items.iter().filter(|i| matches!(i, Item::Widget { parent: Some(p), .. } if p == parent)).count();
                        if one_child && siblings > 1 {
                            bail!("{span}: {parent} is a {}, which holds one widget, and it is given {siblings}", short(&parent_class));
                        }
                        Some(slot_class.to_string())
                    }
                    (None, Some(_)) => Some(CANVAS_SLOT.to_string()),
                    (None, None) => {
                        let roots = file.items.iter().filter(|i| matches!(i, Item::Widget { parent: None, .. })).count();
                        if roots > 1 {
                            bail!("{span}: {name} has no place, so it is the tree's root, and a root is the only widget outside the others; place each widget with `at x y size w h`, or put it inside another");
                        }
                        None
                    }
                };
                let mut defaults = Vec::new();
                for (field, value) in fields {
                    let Some(property) = db.editable_property(class, field) else {
                        bail!("{}: {} has no {field}", value.span, short(class));
                    };
                    defaults.push(ComponentField { name: field.clone(), value: literal(value, &property.ty, &program)? });
                }
                let mut slot_defaults = Vec::new();
                for (field, value) in slot {
                    let Some(slot_class) = &slot_class else {
                        bail!("{}: {name} is the tree's root, which no slot holds", value.span);
                    };
                    let Some(property) = db.editable_property(slot_class, field) else {
                        bail!("{}: {} has no {field}", value.span, short(slot_class));
                    };
                    slot_defaults.push(ComponentField { name: field.clone(), value: literal(value, &property.ty, &program)? });
                }
                ops.push(Op::Widget {
                    name: name.clone(),
                    class: class.clone(),
                    fields: defaults,
                    parent: parent.clone(),
                    place: *place,
                    slot: slot_defaults,
                });
                widget_classes.insert(name.clone(), class.clone());
                program.members.insert(name.clone(), ty);
            }
            Item::Fn { name, params, ret, span, pure, .. } => {
                let pure = *pure != Purity::Impure;
                let mut seen = HashSet::new();
                if let Some(p) = params.iter().find(|p| !seen.insert(&p.name)) {
                    bail!("{span}: {name} has two parameters called {}", p.name);
                }
                let defaults = program.defaults(params)?;
                let outputs = program.outputs(params)?;
                let params = program.params(params)?;
                let ret = ret.as_ref().map(|t| program.ty(t)).transpose()?;
                if ret.is_some() && outputs.iter().any(|(n, _)| n == RETURN_PIN) {
                    bail!("{span}: {name} returns a value through its {RETURN_PIN} pin, so an output cannot be called {RETURN_PIN}");
                }
                if program.members.contains_key(name) || program.functions.contains_key(name) || program.dispatchers.contains_key(name) {
                    bail!("{span}: {name} is declared twice");
                }
                program.inherited(name, *span)?;
                program.functions.insert(name.clone(), UserFunction { params, pure, defaults, ret, outputs });
            }
            Item::Event { name, params, span, .. } => {
                if library {
                    bail!("{span}: a function library has no events");
                }
                if program.members.contains_key(name) || program.functions.contains_key(name) || program.dispatchers.contains_key(name) {
                    bail!("{span}: {name} is declared twice");
                }
                program.inherited(name, *span)?;
                let params = program.params(params)?;
                let defaults = vec![None; params.len()];
                program.functions.insert(name.clone(), UserFunction { params, pure: false, defaults, ret: None, outputs: Vec::new() });
            }
            Item::On { component: Some(component), event, params, ret, span, .. } => {
                let Some(Ty::Object(class)) = program.members.get(component).cloned() else {
                    bail!("{span}: {component} is not a component of this Blueprint");
                };
                let Some(declared) = db.dispatcher(&class, event).map_err(|e| anyhow!("{span}: {e}"))? else {
                    bail!("{span}: {} has no dispatcher {event}", short(&class));
                };
                if let Some(t) = ret {
                    bail!("{}: a dispatcher's handler returns nothing", t.span);
                }
                if let Some(p) = params.iter().find(|p| p.out) {
                    bail!("{span}: a dispatcher's handler gives nothing back, so {} cannot be out", p.name);
                }
                if params.len() != declared.len() || params.iter().any(|p| p.by_ref) {
                    let names: Vec<String> = declared.iter().map(|(n, t)| format!("{n}: {t}")).collect();
                    bail!("{span}: {component}.{event} passes ({}); name each of them, as values", names.join(", "));
                }
                let mut typed = Vec::new();
                for (param, (declared_name, declared_ty)) in params.iter().zip(&declared) {
                    let ty = program.ty(&param.ty)?;
                    if !db.assignable(declared_ty, &ty) {
                        bail!("{span}: {component}.{event}'s {declared_name} is {declared_ty}, not {ty}");
                    }
                    typed.push((param.name.clone(), ty, false));
                }
                let name = bound_function(component, event);
                let defaults = vec![None; typed.len()];
                if program.functions.insert(name, UserFunction { params: typed, pure: false, defaults, ret: None, outputs: Vec::new() }).is_some() {
                    bail!("{span}: there are two handlers for {component}.{event}");
                }
            }
            Item::On { event, params, ret, span, .. } => {
                if library {
                    bail!("{span}: a function library has no events");
                }
                if let Some(p) = params.iter().find(|p| p.by_ref) {
                    bail!("{span}: an event's parameters are values, so {} cannot be ref", p.name);
                }
                let declared = program.event(event).map_err(|e| anyhow!("{span}: {e}"))?;
                // An interface's function may give back outputs, which the handler sets by their
                // names as `out` parameters; any other event's parameters are what it passes.
                let (inputs, outs): (Vec<&ParamDecl>, Vec<&ParamDecl>) = params.iter().partition(|p| !p.out);
                if declared.outputs.is_empty() {
                    if let Some(p) = outs.first() {
                        bail!("{span}: an event's parameters are what it passes, so {} cannot be out; what a handler gives back is its `-> Type`", p.name);
                    }
                }
                if inputs.len() != declared.params.len() {
                    let names: Vec<String> = declared.params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
                    bail!("{span}: {event} passes ({}); name each of them", names.join(", "));
                }
                let mut outputs = Vec::new();
                for (name, wanted) in &declared.outputs {
                    let Some(p) = outs.iter().find(|p| p.name == *name) else {
                        bail!("{span}: {event} gives back {name}: {wanted}; write it as `out {name}: {wanted}`");
                    };
                    let ty = program.ty(&p.ty)?;
                    if ty != *wanted {
                        bail!("{span}: {event}'s {name} is {wanted}, not {ty}");
                    }
                    outputs.push((name.clone(), ty));
                }
                if let Some(p) = outs.iter().find(|p| !declared.outputs.iter().any(|(n, _)| *n == p.name)) {
                    bail!("{span}: {event} gives back no {}", p.name);
                }
                let mut typed = Vec::new();
                for (param, (event_param, event_ty)) in inputs.iter().zip(&declared.params) {
                    let ty = program.ty(&param.ty)?;
                    if !db.assignable(event_ty, &ty) {
                        bail!("{span}: {event}'s {event_param} is {event_ty}, not {ty}");
                    }
                    typed.push((param.name.clone(), ty, false));
                }
                let ret = match (ret, &declared.ret) {
                    (None, None) => None,
                    (Some(t), Some(wanted)) => {
                        let ty = program.ty(t)?;
                        if ty != *wanted {
                            bail!("{span}: {event} returns {wanted}, not {ty}");
                        }
                        Some(ty)
                    }
                    (None, Some(wanted)) => bail!("{span}: {event} returns {wanted}; write `-> {wanted}`"),
                    (Some(_), None) => bail!("{span}: {event} returns nothing"),
                };
                // An event that returns a value, or is const, is overridden as a function of its own
                // name.
                let name = if declared.function {
                    let names: Vec<String> = declared.params.iter().map(|(n, _)| n.clone()).collect();
                    overrides.insert(event.clone(), (declared.owner.clone(), names));
                    event.clone()
                } else {
                    event_function(event)
                };
                let defaults = vec![None; typed.len()];
                if program.functions.insert(name, UserFunction { params: typed, pure: false, defaults, ret, outputs }).is_some() {
                    bail!("{span}: there are two handlers for {event}");
                }
            }
        }
    }

    // Each body is lowered on its own, since a function's locals are declared with it but only
    // known once its body has been read. Every function is declared before any body, so that a
    // call to one resolves wherever it is.
    let mut declarations = Vec::new();
    let mut bodies = Vec::new();
    let mut events = Vec::new();
    let mut bound = Vec::new();
    for item in &file.items {
        let (name, body, description, handler, access, purity) = match item {
            Item::Fn { name, body, doc, access, pure, .. } => (name.clone(), body, doc.clone(), None, *access, *pure),
            // A custom event is always in the event graph, where it may wait.
            Item::Event { name, body, .. } => {
                let (variables, nodes) = waiting_handler(&program, &Handler::Custom(name.clone()), name, body)?;
                declarations.extend(variables);
                bodies.extend(nodes);
                continue;
            }
            Item::On { component: Some(component), event, body, .. } => {
                let handler = Handler::Bound(component.clone(), event.clone());
                (bound_function(component, event), body, None, Some(handler), Access::Public, Purity::Impure)
            }
            Item::On { event, body, .. } if overrides.contains_key(event) => (event.clone(), body, None, None, Access::Public, Purity::Impure),
            Item::On { event, body, .. } => {
                (event_function(event), body, None, Some(Handler::Event(event.clone())), Access::Public, Purity::Impure)
            }
            Item::Var { .. }
            | Item::Widget { .. }
            | Item::Category { .. }
            | Item::Use { .. }
            | Item::Implements { .. }
            | Item::Dispatcher { .. }
            | Item::Component { .. }
            | Item::Default { .. } => continue,
        };
        let function = &program.functions[&name];
        let mut graph = match overrides.get(&name) {
            // An override's entry node names its parameters as the event does, so the names the
            // source gives them read those pins.
            Some((_, entry_names)) => {
                let mut graph = Graph::new(&name, &program, HashMap::new(), function.ret.clone());
                for ((param, ty, _), entry_name) in function.params.iter().zip(entry_names) {
                    let value = Value { source: Source::Pin(pin(&graph.id("entry"), entry_name)), ty: ty.clone() };
                    graph.aliases.insert(param.clone(), value);
                }
                graph
            }
            None => {
                let params = function.params.iter().map(|(n, t, _)| (n.clone(), t.clone())).collect();
                Graph::new(&name, &program, params, function.ret.clone())
            }
        };
        graph.handler = handler.is_some();
        graph.checks_effects = purity == Purity::Pure;
        graph.declare_outputs(&function.outputs);
        let lowered = graph.function_body(body);
        if let (Err(_), true, Some(handler)) = (&lowered, graph.waits, &handler) {
            let (variables, nodes) = waiting_handler(&program, handler, &name, body)?;
            declarations.extend(variables);
            bodies.extend(nodes);
            continue;
        }
        lowered?;
        match handler {
            Some(Handler::Event(event)) => events.push(event),
            Some(Handler::Bound(component, dispatcher)) => bound.push((component, dispatcher)),
            Some(Handler::Custom(_)) => unreachable!("a custom event is lowered into the event graph above"),
            None => {}
        }
        declarations.push(Op::Function {
            name: name.clone(),
            params: function
                .params
                .iter()
                .zip(&function.defaults)
                .map(|((n, t, by_ref), default)| Ok(Param { name: n.clone(), ty: t.op_type()?, by_ref: *by_ref, default: default.clone() }))
                .collect::<Result<_>>()?,
            // The return value's pin first, then a pin per `out` parameter, in the order declared.
            returns: function
                .ret
                .iter()
                .map(|t| (RETURN_PIN, t))
                .chain(function.outputs.iter().map(|(n, t)| (n.as_str(), t)))
                .map(|(n, t)| Ok(Param { name: n.to_string(), ty: t.op_type()?, by_ref: false, default: None }))
                .collect::<Result<_>>()?,
            locals: graph.locals_declared()?,
            description,
            category: category.clone(),
            overrides: overrides.get(&name).map(|(owner, _)| owner.clone()),
            pure: purity != Purity::Impure,
            access: access_word(access).map(String::from),
        });
        bodies.extend(graph.ops);
    }
    // A function bound to an engine delegate that passes 32-bit floats takes them as 32-bit
    // floats, as the editor makes a function to bind to one, so Blueprint binds the function
    // itself: otherwise it binds a proxy of its own for each delegate made, which no unbind matches.
    for op in &mut declarations {
        let Op::Function { name, params, returns, .. } = op else { continue };
        let Some(narrowed) = program.narrowed.borrow_mut().remove(name.as_str()) else { continue };
        for (param, floats) in params.iter_mut().chain(returns.iter_mut()).zip(narrowed.widths) {
            if floats == Floats::Single && param.ty == crate::ops::Type::Float {
                param.ty = crate::ops::Type::Float32;
            }
        }
    }
    ops.extend(declarations);
    ops.extend(bodies);

    let mut graph = Graph::new(EVENT_GRAPH, &program, HashMap::new(), None);
    let handlers = events.into_iter().map(Handler::Event).chain(bound.into_iter().map(|(c, d)| Handler::Bound(c, d)));
    for handler in handlers {
        let (node, pins) = handler.node(&mut graph)?;
        let name = handler.function();
        let call = graph.node("call", NodeKind::Call { function: format!("self:{name}") });
        graph.link(pin(&node, "then"), pin(&call, "execute"));
        for (event_param, (param, _, _)) in pins.iter().zip(&program.functions[&name].params) {
            graph.link(pin(&node, event_param), pin(&call, param));
        }
    }
    ops.extend(graph.ops);

    ops.push(Op::Compile);
    // The class defaults, on the class the compile made, as the Class Defaults panel sets them.
    let mut defaulted = HashSet::new();
    for item in &file.items {
        if let Item::Default { path, value, span } = item {
            if !defaulted.insert(path.join(".")) {
                bail!("{span}: {} is given a default twice", path.join("."));
            }
            ops.push(class_default(&program, path, value, *span)?);
        }
    }
    ops.push(Op::Save);
    let doc = Document { version: VERSION, ops };
    doc.check()?;
    Ok(doc)
}

impl Program<'_> {
    /// Fails when the parent class already has a member of this name, which the Blueprint
    /// compiler would refuse, or worse, take for the parent's.
    fn inherited(&self, name: &str, span: Span) -> Result<()> {
        if self.db.property(&self.parent, name).is_some() {
            bail!("{span}: {} already has a member {name}; pick another name", short(&self.parent));
        }
        if self.db.function(&self.parent, name)?.is_some() {
            bail!("{span}: {} already has a function {name}; pick another name", short(&self.parent));
        }
        if let Some(interface) = self.interfaces.iter().find(|i| self.db.interface_declares(i, name)) {
            bail!("{span}: {} has a function {name}; implement it with `on {name}`, or pick another name", short(interface));
        }
        Ok(())
    }

    /// The interface an `implements` names, checked: one a Blueprint may implement, and not one it
    /// implements already, by an earlier line or through its parent.
    fn implemented(&self, interface: &TypeName, span: Span, earlier: &[String]) -> Result<String> {
        if self.db.is_subclass(&self.parent, FUNCTION_LIBRARY) {
            bail!("{span}: a function library implements no interfaces");
        }
        let Ty::Interface(path) = self.ty(interface)? else {
            bail!("{span}: {} is not an interface", interface.name);
        };
        if !self.db.is_implementable(&path) {
            bail!("{span}: {} is an interface a Blueprint cannot implement", short(&path));
        }
        if earlier.contains(&path) {
            bail!("{span}: {} is implemented twice", short(&path));
        }
        // Unreal would let it, but would give it empty graphs for the functions that return a
        // value, in place of the ones the parent has.
        if self.db.interfaces_of(&self.parent).contains(&path) {
            bail!(
                "{span}: {} already implements {}, and so does this Blueprint; implement its functions with `on`, without `implements`",
                short(&self.parent),
                short(&path)
            );
        }
        Ok(path)
    }

    /// An event a handler may be written for: one the parent declares, or else one of an interface
    /// the Blueprint implements.
    fn event(&self, name: &str) -> Result<Event> {
        if self.interfaces.is_empty() || self.db.has_function(&self.parent, name) {
            return self.db.event(&self.parent, name);
        }
        let declaring: Vec<&String> = self.interfaces.iter().filter(|i| self.db.interface_declares(i, name)).collect();
        match declaring.as_slice() {
            [] => {
                let names: Vec<&str> = self.interfaces.iter().map(|i| short(i)).collect();
                bail!("{} has no event {name}, and neither has any interface it implements: {}", short(&self.parent), names.join(", "))
            }
            [interface] => self.db.event(interface, name),
            many => {
                let names: Vec<&str> = many.iter().map(|i| short(i)).collect();
                bail!("{name} is a function of {}, which a Blueprint cannot implement for each", names.join(" and "))
            }
        }
    }

    /// The interface, of those an object of `class` implements, that has a function of this name.
    fn interface_with(&self, class: &str, name: &str) -> Result<Option<String>> {
        // This Blueprint's own are known before its class is.
        let interfaces = if self.class.as_deref() == Some(class) { self.interfaces.clone() } else { self.db.interfaces_of(class) };
        let declaring: Vec<String> = interfaces.into_iter().filter(|i| self.db.interface_declares(i, name)).collect();
        match declaring.as_slice() {
            [] => Ok(None),
            [interface] => Ok(Some(interface.clone())),
            many => {
                let names: Vec<&str> = many.iter().map(|i| short(i)).collect();
                bail!("{name} is a function of {}; cast to the one to call it through", names.join(" and "))
            }
        }
    }

    /// The inputs among the parameters: each one's name, type and whether it is a ref. The `out`
    /// parameters are skipped, since a call does not pass them.
    fn params(&self, params: &[ParamDecl]) -> Result<Vec<(String, Ty, bool)>> {
        params.iter().filter(|p| !p.out).map(|p| Ok((p.name.clone(), self.ty(&p.ty)?, p.by_ref))).collect()
    }

    /// Each input's default, as text, as a pin holds it.
    fn defaults(&self, params: &[ParamDecl]) -> Result<Vec<Option<String>>> {
        params.iter().filter(|p| !p.out).map(|p| p.default.as_ref().map(|e| literal(e, &self.ty(&p.ty)?, self)).transpose()).collect()
    }

    /// The `out` parameters among the parameters: each one's name and type.
    fn outputs(&self, params: &[ParamDecl]) -> Result<Vec<(String, Ty)>> {
        params.iter().filter(|p| p.out).map(|p| Ok((p.name.clone(), self.ty(&p.ty)?))).collect()
    }

    fn ty(&self, t: &TypeName) -> Result<Ty> {
        let ty = self.any_ty(t)?;
        // Blueprint has no container of delegates.
        match &ty {
            Ty::Array(inner) | Ty::Set(inner) if inner.has_delegate() => bail!("{}: a container cannot hold a delegate", t.span),
            Ty::Map(key, value) if key.has_delegate() || value.has_delegate() => bail!("{}: a container cannot hold a delegate", t.span),
            _ => Ok(ty),
        }
    }

    fn any_ty(&self, t: &TypeName) -> Result<Ty> {
        let named = |name: &str| TypeName { name: name.to_string(), span: t.span };
        if let Some(inner) = t.name.strip_prefix("array<").and_then(|r| r.strip_suffix('>')) {
            let inner = self.ty(&named(inner))?;
            if matches!(inner, Ty::Array(_) | Ty::Map(..)) {
                bail!("{}: an array cannot hold {inner}; Blueprints have no nested containers", t.span);
            }
            return Ok(Ty::Array(Box::new(inner)));
        }
        if let Some(inner) = t.name.strip_prefix("set<").and_then(|r| r.strip_suffix('>')) {
            let inner = self.ty(&named(inner))?;
            if matches!(inner, Ty::Array(_) | Ty::Set(_) | Ty::Map(..)) {
                bail!("{}: a set cannot hold {inner}; Blueprints have no nested containers", t.span);
            }
            return Ok(Ty::Set(Box::new(inner)));
        }
        if let Some(inner) = t.name.strip_prefix("map<").and_then(|r| r.strip_suffix('>')) {
            let Some((key, value)) = crate::types::split_pair(inner) else { bail!("{}: a map is map<Key, Value>", t.span) };
            let (key, value) = (self.ty(&named(key))?, self.ty(&named(value))?);
            for ty in [&key, &value] {
                if matches!(ty, Ty::Array(_) | Ty::Map(..)) {
                    bail!("{}: a map cannot hold {ty}; Blueprints have no nested containers", t.span);
                }
            }
            return Ok(Ty::Map(Box::new(key), Box::new(value)));
        }
        if let Some(inner) = t.name.strip_prefix("class<").and_then(|r| r.strip_suffix('>')) {
            let Ty::Object(class) = self.ty(&named(inner))? else { bail!("{}: class<T> takes a class, not {inner}", t.span) };
            return Ok(Ty::Class(class));
        }
        if let Some(inner) = t.name.strip_prefix("soft<").and_then(|r| r.strip_suffix('>')) {
            return match self.ty(&named(inner))? {
                Ty::Object(class) => Ok(Ty::SoftObject(class)),
                Ty::Class(class) => Ok(Ty::SoftClass(class)),
                other => bail!("{}: soft<T> refers to an object or a class<T>, not {other}", t.span),
            };
        }
        if let Some(ty) = builtins::plain_type(&t.name) {
            return Ok(ty);
        }
        let name = t.name.as_str();
        Ok(if let Some(path) = self.class(name).map_err(|e| anyhow!("{}: {e}", t.span))? {
            if self.db.is_interface(&path) { Ty::Interface(path) } else { Ty::Object(path) }
        } else if let Some(path) = self.structure(name).map_err(|e| anyhow!("{}: {e}", t.span))? {
            Ty::Struct(path)
        } else if let Some(path) = self.enumeration(name).map_err(|e| anyhow!("{}: {e}", t.span))? {
            Ty::Enum(path)
        } else if let Some(path) = self.db.delegate(name, &self.scope()).map_err(|e| anyhow!("{}: {e}", t.span))? {
            Ty::Delegate(path)
        } else {
            bail!("{}: unknown type '{name}'", t.span)
        })
    }

    /// A class by name, as this Blueprint may use it.
    fn class(&self, name: &str) -> Result<Option<String>> {
        self.db.class_in(name, self.editor, &self.scope())
    }

    /// Where this Blueprint looks for a type it names by its short name.
    fn scope(&self) -> TypeScope<'_> {
        TypeScope { folder: &self.folder, uses: &self.uses }
    }

    /// A struct by name, as this Blueprint finds it.
    fn structure(&self, name: &str) -> Result<Option<String>> {
        self.db.structure(name, &self.scope())
    }

    /// An enum by name, as this Blueprint finds it.
    fn enumeration(&self, name: &str) -> Result<Option<String>> {
        self.db.enumeration(name, &self.scope())
    }

    /// A function of a class this Blueprint calls, which it may call: a Blueprint's function that
    /// is private only from the class that declares it, and one that is protected only from that
    /// class and those that derive from it, as Blueprint allows.
    fn function(&self, class: &str, name: &str) -> Result<Option<Function>> {
        let found = self.db.function_in(class, name, if self.editor { &["editor_only"] } else { &[] })?;
        // Blueprint ignores the access of an engine function, whose flags often only hide it
        // from other C++.
        if let Some(function) = found.as_ref().filter(|f| !f.owner.starts_with("/Script/")) {
            let caller = self.caller.as_deref();
            match function.access {
                Some("private") if caller != Some(function.owner.as_str()) => {
                    bail!("{name} is private to {}", blueprint_name(&function.owner));
                }
                Some("protected") if !caller.is_some_and(|c| self.db.is_subclass(c, &function.owner)) => {
                    bail!("{name} is protected, so only {} and the Blueprints that derive from it can call it", blueprint_name(&function.owner));
                }
                _ => {}
            }
        }
        Ok(found)
    }

    fn library(&self, class: &str, name: &str) -> Result<Function> {
        self.library_in(class, name, &[])
    }

    fn library_in(&self, class: &str, name: &str, allowed: &[&str]) -> Result<Function> {
        self.db
            .function_in(class, name, allowed)?
            .ok_or_else(|| anyhow!("the types have no {class}:{name}; is the dump from another engine?"))
    }
}

/// What a name stands for where a delegate is expected.
enum Callee {
    /// A function, with the object whose function it is when it is not this one's, and its
    /// parameters and return value in order. `own` is one this Blueprint declares, whose floats
    /// loom makes as wide as a delegate passes them.
    Function { name: String, object: Option<Source>, shape: Vec<Slot>, own: bool },
    /// Anything else, lowered as any value is.
    Value(Value),
}

/// Whether a function takes and returns what a delegate's signature passes it, as Blueprint
/// compares them: the same types, in order, each passed the same way, and the same return value.
/// Names don't matter. With `widths`, each float is as wide, 32-bit or 64-bit, which a function
/// loom declares need not be, since loom makes its floats as wide as the signature's.
fn same_shape(function: &[Slot], signature: &[Slot], widths: bool) -> bool {
    function.len() == signature.len()
        && function.iter().zip(signature).all(|(f, s)| f.ty == s.ty && f.passing == s.passing && (!widths || f.floats == s.floats))
}

/// A function's parameters and return value as a message shows them: fn(a: int, b: ref Vector) -> bool.
/// A 32-bit float is a float32.
fn describe(shape: &[Slot]) -> String {
    let ty = |s: &Slot| if s.floats == Floats::Single { "float32".to_string() } else { s.ty.to_string() };
    let params: Vec<String> = shape
        .iter()
        .filter_map(|s| match s.passing {
            Passing::Value => Some(format!("{}: {}", s.name, ty(s))),
            Passing::Ref => Some(format!("{}: ref {}", s.name, ty(s))),
            Passing::Out => Some(format!("out {}: {}", s.name, ty(s))),
            Passing::Return => None,
        })
        .collect();
    match shape.iter().find(|s| s.passing == Passing::Return) {
        Some(ret) => format!("fn({}) -> {}", params.join(", "), ty(ret)),
        None => format!("fn({})", params.join(", ")),
    }
}

/// An `on` handler that is an event node in the event graph, rather than a function the Blueprint
/// overrides.
enum Handler {
    Event(String),
    /// A custom event of the Blueprint's own, by its name.
    Custom(String),
    /// A component's dispatcher, bound as the component's green + in the Details panel binds it.
    Bound(String, String),
}

impl Handler {
    /// The function the handler is lowered into, unless it waits.
    fn function(&self) -> String {
        match self {
            Handler::Event(event) => event_function(event),
            Handler::Custom(name) => name.clone(),
            Handler::Bound(component, dispatcher) => bound_function(component, dispatcher),
        }
    }

    /// Its event node, in `graph`, and the names of the node's pins that carry the handler's
    /// parameters.
    fn node(&self, graph: &mut Graph) -> Result<(String, Vec<String>)> {
        let program = graph.program;
        Ok(match self {
            Handler::Event(event) => {
                // An interface's event is named by the interface, and the parent's by the parent.
                let declared = program.event(event)?;
                let class = if program.db.is_interface(&declared.owner) { declared.owner.clone() } else { program.parent.clone() };
                let node = graph.node("event", NodeKind::Event { event: event.clone(), class });
                (node, declared.params.into_iter().map(|(n, _)| n).collect())
            }
            Handler::Custom(name) => {
                let params: Vec<Param> = program.functions[name]
                    .params
                    .iter()
                    .map(|(n, t, _)| Ok(Param { name: n.clone(), ty: t.op_type()?, by_ref: false, default: None }))
                    .collect::<Result<_>>()?;
                let pins = params.iter().map(|p| p.name.clone()).collect();
                (graph.node("event", NodeKind::CustomEvent { name: name.clone(), params }), pins)
            }
            Handler::Bound(component, dispatcher) => {
                let node = graph.node("bound", NodeKind::BoundEvent { component: component.clone(), dispatcher: dispatcher.clone() });
                let Some(Ty::Object(class)) = program.members.get(component) else { unreachable!() };
                let declared = program.db.dispatcher(class, dispatcher)?.unwrap_or_default();
                (node, declared.into_iter().map(|(n, _)| n).collect())
            }
        })
    }
}

/// An `on` handler that waits, lowered into the event graph, which is where a Blueprint can wait:
/// its event node starts the chain rather than calling a function, its parameters are read from
/// that node, and its locals are variables of the Blueprint, since the event graph has none. Gives
/// those variables, then the nodes.
fn waiting_handler(program: &Program, handler: &Handler, name: &str, body: &[Stmt]) -> Result<(Vec<Op>, Vec<Op>)> {
    let mut graph = Graph::new(name, program, HashMap::new(), None);
    graph.graph = EVENT_GRAPH.to_string();
    let (node, pins) = handler.node(&mut graph)?;
    for (event_param, (param, ty, _)) in pins.iter().zip(&program.functions[name].params) {
        graph.aliases.insert(param.clone(), Value { source: Source::Pin(pin(&node, event_param)), ty: ty.clone() });
    }
    changed_in(body, &mut graph.changed, &|name| changes_container(program.db, name));
    graph.cursor = vec![pin(&node, "then")];
    graph.block(body)?;

    // A local's get and set nodes become the variable's, whose pin is named after it. A custom
    // event's name is the source's own, so its variables take loom's prefix, as a handler's do.
    let prefix = if name.starts_with(RESERVED_PREFIX) { name.to_string() } else { format!("{RESERVED_PREFIX}Event_{name}") };
    let member = |local: &str| format!("{prefix}_{local}");
    let mut renamed = HashMap::new();
    for op in &mut graph.ops {
        if let Op::Node { id, kind: NodeKind::Get { var, scope, .. } | NodeKind::Set { var, scope, .. }, .. } = op {
            if *scope == Scope::Local {
                renamed.insert(id.clone(), var.clone());
                *var = member(var);
                *scope = Scope::SelfMember;
            }
        }
    }
    let rename = |p: &mut PinRef| {
        if renamed.get(&p.node) == Some(&p.pin) {
            p.pin = member(&p.pin);
        }
    };
    for op in &mut graph.ops {
        match op {
            Op::Link { from, to } => {
                rename(from);
                rename(to);
            }
            Op::Default { pin, .. } => rename(pin),
            _ => {}
        }
    }
    let variables = graph
        .locals
        .iter()
        .map(|(local, ty)| {
            Ok(Op::Variable {
                name: member(local),
                ty: ty.op_type()?,
                default: None,
                editable: false,
                // A handler's own state, which nothing else has any business reading.
                private: true,
                description: None,
                category: None,
            })
        })
        .collect::<Result<_>>()?;
    Ok((variables, graph.ops))
}

fn event_function(event: &str) -> String {
    format!("{RESERVED_PREFIX}On_{event}")
}

fn bound_function(component: &str, dispatcher: &str) -> String {
    format!("{RESERVED_PREFIX}On_{component}_{dispatcher}")
}

/// The local that holds an `out` parameter while the function runs. It is not named after the
/// parameter, since the result node's pin of that name is a property of the function too.
fn out_local(param: &str) -> String {
    format!("{RESERVED_PREFIX}Out_{param}")
}

fn pin(node: &str, pin: &str) -> PinRef {
    PinRef { node: node.to_string(), pin: pin.to_string() }
}

fn short(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}

/// A Blueprint's class by the Blueprint's name, as source calls it: BP_X for /Game/BP_X.BP_X_C.
fn blueprint_name(path: &str) -> &str {
    let name = short(path);
    name.strip_suffix("_C").unwrap_or(name)
}

/// A literal as a pin default, checked against the type it must have.
/// Why code cannot reach a member of a class: it has none of that name, or only the Details panel
/// sets it.
fn no_member(db: &TypeDb, class: &str, name: &str) -> String {
    if db.editable_property(class, name).is_some_and(|p| p.private) {
        format!("{name} is private to {}, so only it can read or set it", short(db.property_owner(class, name).as_deref().unwrap_or(class)))
    } else if db.editable_property(class, name).is_some() {
        format!("{name} is set only in the Details panel, so code cannot reach it; give it a value with `default`, or in a component's fields")
    } else {
        format!("{} has no {name}", short(class))
    }
}

/// `default a.b = value`: a property the parent declares, or a field inside one, and the value to
/// give it, written out or copied from another class's default.
fn class_default(program: &Program, path: &[String], value: &Expr, span: Span) -> Result<Op> {
    let db = program.db;
    let (first, rest) = path.split_first().expect("a default names a property");
    if program.members.contains_key(first) {
        bail!("{span}: {first} is this Blueprint's own; give it its default where it is declared");
    }
    let Some(property) = db.editable_property(&program.parent, first) else {
        bail!("{span}: {} has no {first} to give a default", short(&program.parent));
    };
    let mut ty = property.ty;
    for (i, field) in rest.iter().enumerate() {
        let Ty::Struct(structure) = &ty else { bail!("{span}: {} is not a struct, so it has no {field}", path[..=i].join(".")) };
        let Some(field_ty) = db.editable_field(structure, field) else {
            bail!("{span}: {} has no field {field}", short(structure));
        };
        ty = field_ty;
    }
    let name = path.join(".");
    // `Class.Name`: that class's own default for the same property, copied whole, so fields no
    // literal can reach come along too.
    if let ExprKind::Member(object, member) = &value.kind {
        if let ExprKind::Name(class_name) = &object.kind {
            let class = program.ty(&TypeName { name: class_name.clone(), span: value.span }).ok();
            if let Some(Ty::Object(class) | Ty::Class(class)) = class {
                let Some(source) = db.editable_property(&class, member) else {
                    bail!("{}: {} has no {member}", value.span, short(&class));
                };
                let source_ty = source.ty;
                if !rest.is_empty() || member != first {
                    bail!("{}: a default is copied from the same property of another class, {class_name}.{first}", value.span);
                }
                if source_ty != ty {
                    bail!("{}: {class_name}.{member} is a {source_ty}, and {name} a {ty}", value.span);
                }
                return Ok(Op::ClassDefault { name, value: None, from: Some(class) });
            }
        }
    }
    Ok(Op::ClassDefault { name, value: Some(literal(value, &ty, program)?), from: None })
}

fn literal(e: &Expr, ty: &Ty, p: &Program) -> Result<String> {
    let db = p.db;
    // `S()` is the struct with every field at its zero.
    if let (ExprKind::Call(name, args, false), Ty::Struct(_)) = (&e.kind, ty) {
        if args.is_empty() && p.structure(name)?.is_some() {
            return literal(&Expr { kind: ExprKind::Make(name.clone(), Vec::new()), span: e.span }, ty, p);
        }
    }
    Ok(match (&e.kind, ty) {
        // A container is written as Unreal writes one in text, with each item quoted as needed.
        (ExprKind::ArrayLiteral(items), Ty::Array(inner) | Ty::Set(inner)) => {
            let items: Vec<String> = items.iter().map(|i| item_literal(i, inner, p)).collect::<Result<_>>()?;
            format!("({})", items.join(","))
        }
        (ExprKind::EmptyArray, Ty::Array(_) | Ty::Set(_)) | (ExprKind::MapLiteral(_), Ty::Map(..)) if is_empty_literal(e) => String::new(),
        (ExprKind::MapLiteral(entries), Ty::Map(key, value)) => {
            let mut seen = Vec::new();
            let mut pairs = Vec::new();
            for (k, v) in entries {
                let k_text = item_literal(k, key, p)?;
                if seen.contains(&k_text) {
                    bail!("{}: the key {k_text} is in the map twice", k.span);
                }
                pairs.push(format!("({k_text},{})", item_literal(v, value, p)?));
                seen.push(k_text);
            }
            format!("({})", pairs.join(","))
        }
        // A class written by path is checked when the types know it. One they don't, such as a
        // class not loaded when they were saved, is taken as written.
        (ExprKind::Path(path), Ty::Class(base) | Ty::SoftClass(base)) => {
            if db.class_in(path, true, &p.scope()).ok().flatten().is_some() && !db.is_subclass(path, base) {
                bail!("{}: {} is not a {}", e.span, short(path), short(base));
            }
            path.clone()
        }
        (ExprKind::Path(path), Ty::SoftObject(_) | Ty::Object(_)) => path.clone(),
        // A class written by name, as code writes one.
        (ExprKind::Name(name), Ty::Class(base) | Ty::SoftClass(base)) => {
            let Some(class) = db.class_in(name, false, &p.scope()).map_err(|err| anyhow!("{}: {err}", e.span))? else {
                bail!("{}: no class {name}", e.span);
            };
            if !db.is_subclass(&class, base) {
                bail!("{}: {name} is not a {}", e.span, short(base));
            }
            class
        }
        // A colour is written as the hex a screenshot or a colour picker gives, in sRGB.
        (ExprKind::Str(hex), Ty::Struct(path)) if path == LINEAR_COLOR || path == SLATE_COLOR => {
            let colour = linear_colour(hex).map_err(|err| anyhow!("{}: {err}", e.span))?;
            if path == SLATE_COLOR { format!("(SpecifiedColor={colour})") } else { colour }
        }
        (ExprKind::Make(name, fields), Ty::Struct(path)) => {
            // A struct the engine makes with another's make function, such as Slate's stand-in
            // for a Vector2D, is written as that other struct.
            let Some(written) = p.structure(name)? else { bail!("{}: expected a {ty}, not a {name}", e.span) };
            let made_alike = db.native_make(&written).is_some() && db.native_make(&written) == db.native_make(path);
            if written != *path && !made_alike {
                bail!("{}: expected a {ty}, not a {name}", e.span);
            }
            let mut parts = Vec::new();
            for (field, value) in fields {
                let Some(field_ty) = db.editable_field(&written, field) else {
                    bail!("{}: {name} has no field {field} that Blueprints can set", value.span);
                };
                parts.push(format!("{field}={}", field_literal(value, &field_ty, p)?));
            }
            format!("({})", parts.join(","))
        }
        (ExprKind::Member(object, value), Ty::Enum(path)) => {
            let ExprKind::Name(name) = &object.kind else { bail!("{}: expected a {ty} value, such as {ty}.Something", e.span) };
            if p.enumeration(name)?.as_deref() != Some(path.as_str()) {
                bail!("{}: expected a {ty} value, not one of {name}", e.span);
            }
            enum_value(db, path, value, e.span)?
        }
        (ExprKind::ArrayLiteral(_) | ExprKind::EmptyArray | ExprKind::MapLiteral(_) | ExprKind::Path(_), _) => {
            bail!("{}: expected a {ty} literal", e.span)
        }
        // A number written with a minus is still a number written out.
        (ExprKind::Unary(UnaryOp::Neg, inner), Ty::Int | Ty::Int64 | Ty::Float)
            if matches!(inner.kind, ExprKind::Int(_) | ExprKind::Float(_)) =>
        {
            format!("-{}", literal(inner, ty, p)?)
        }
        (ExprKind::Int(n), Ty::Int | Ty::Int64) => n.to_string(),
        (ExprKind::Int(n), Ty::Byte) => {
            if !(0..=255).contains(n) {
                bail!("{}: a byte is 0 to 255, not {n}", e.span);
            }
            n.to_string()
        }
        (ExprKind::Int(n), Ty::Float) => format!("{n}.0"),
        (ExprKind::Float(n), Ty::Float) => n.to_string(),
        (ExprKind::Bool(b), Ty::Bool) => b.to_string(),
        (ExprKind::Str(s), Ty::Str | Ty::Name | Ty::Text) => s.clone(),
        (ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_), _) => {
            bail!("{}: expected a {ty} literal", e.span)
        }
        _ => bail!("{}: a value set here must be written out, such as 5, \"text\" or Vector(X: 1, Y: 2, Z: 3)", e.span),
    })
}

/// `#rrggbb` or `#rrggbbaa`, in sRGB as screens show it, as the linear colour Unreal holds.
fn linear_colour(hex: &str) -> Result<String> {
    let digits = hex.strip_prefix('#').filter(|d| matches!(d.len(), 6 | 8) && d.chars().all(|c| c.is_ascii_hexdigit()));
    let Some(digits) = digits else { bail!("a colour is written #rrggbb or #rrggbbaa, not {hex}") };
    let byte = |i: usize| f64::from(u8::from_str_radix(&digits[i..i + 2], 16).unwrap()) / 255.0;
    let linear = |c: f64| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    let alpha = if digits.len() == 8 { byte(6) } else { 1.0 };
    Ok(format!("(R={:.6},G={:.6},B={:.6},A={:.6})", linear(byte(0)), linear(byte(2)), linear(byte(4)), alpha))
}

/// One of an enum's values, by name, as its pin holds it.
fn enum_value(db: &TypeDb, path: &str, value: &str, span: Span) -> Result<String> {
    let values = db.enum_values(path);
    if !values.iter().any(|v| v == value) {
        bail!("{span}: {} has no value {value}; it has {}", short(path), values.join(", "));
    }
    Ok(value.to_string())
}

/// The type of an array written out in plain literals of one type, where ints among floats are
/// floats: [1, 2] is an array<int>, [1, 2.5] an array<float>.
fn literal_array_type(items: &[Expr]) -> Option<Ty> {
    let mut item = None;
    for e in items {
        let ty = match e.kind {
            ExprKind::Int(_) => Ty::Int,
            ExprKind::Float(_) => Ty::Float,
            ExprKind::Str(_) => Ty::Str,
            ExprKind::Bool(_) => Ty::Bool,
            _ => return None,
        };
        item = match (item, ty) {
            (None, t) => Some(t),
            (Some(Ty::Int), Ty::Float) | (Some(Ty::Float), Ty::Int) => Some(Ty::Float),
            (Some(a), b) if a == b => Some(a),
            _ => return None,
        };
    }
    Some(Ty::Array(Box::new(item?)))
}

fn is_empty_literal(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::EmptyArray => true,
        ExprKind::MapLiteral(entries) => entries.is_empty(),
        _ => false,
    }
}

/// A literal inside a container's default, where strings, names and paths are quoted.
fn item_literal(e: &Expr, ty: &Ty, p: &Program) -> Result<String> {
    if let Ty::Array(_) | Ty::Set(_) | Ty::Map(..) = ty {
        bail!("{}: Blueprints have no nested containers", e.span);
    }
    field_literal(e, ty, p)
}

/// A literal for a struct's field, quoted as a container's item is. A field may be a container.
fn field_literal(e: &Expr, ty: &Ty, p: &Program) -> Result<String> {
    let text = literal(e, ty, p)?;
    Ok(match ty {
        Ty::Str | Ty::Name | Ty::SoftObject(_) | Ty::SoftClass(_) | Ty::Object(_) | Ty::Class(_) => format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\"")),
        Ty::Text => format!("INVTEXT(\"{}\")", text.replace('\\', "\\\\").replace('"', "\\\"")),
        _ => text,
    })
}

/// A loop being lowered: the exec outputs `continue` and `break` left its body from, which join
/// its next pass and what follows it once the body is lowered.
#[derive(Default)]
struct Loop {
    /// The name a labelled loop is given, which a break or continue can name it by.
    label: Option<String>,
    continues: Vec<PinRef>,
    breaks: Vec<PinRef>,
}

struct Graph<'a> {
    name: String,
    /// The graph its nodes go in: the function's own, or the event graph for an `on` handler that
    /// waits.
    graph: String,
    /// An `on` handler's body, lowered as a function until it calls something that waits.
    handler: bool,
    /// Set when a handler lowered as a function calls something that waits, which only the event
    /// graph can do.
    waits: bool,
    program: &'a Program<'a>,
    params: HashMap<String, Ty>,
    ret: Option<Ty>,
    /// The function's `out` parameters, in the order declared. Each is held in a local of its
    /// own, named by [`out_local`], which every return feeds to the result node's pin of the
    /// parameter's name.
    outputs: Vec<(String, Ty)>,
    locals: Vec<(String, Ty)>,
    /// Names bound by `let` to a call's outputs, which are read from the call node itself.
    aliases: HashMap<String, Value>,
    /// Names a `let` bound to a struct, array or map read where it lies rather than copied.
    views: HashSet<String>,
    /// Every path the function assigns to or changes in place, such as `a.b[]`, which a view must
    /// not overlap.
    changed: Vec<Vec<String>>,
    ops: Vec<Op>,
    count: usize,
    /// The exec outputs the next statement continues from. More than one after an if, and none
    /// after a return, a break or a continue.
    cursor: Vec<PinRef>,
    /// The loops the statement being lowered is in, the innermost last.
    loops: Vec<Loop>,
    /// The label of the loop about to be lowered, which it takes when it starts.
    label: Option<String>,
    /// While lowering a loop's condition, the first node it put on the chain, which is where each
    /// pass of the loop has to start again.
    head: Option<Option<String>>,
    /// A `pure fn`'s body, which may change nothing outside the function.
    checks_effects: bool,
}

impl<'a> Graph<'a> {
    fn new(name: &str, program: &'a Program<'a>, params: HashMap<String, Ty>, ret: Option<Ty>) -> Self {
        Graph {
            checks_effects: false,
            name: name.to_string(),
            graph: name.to_string(),
            handler: false,
            waits: false,
            program,
            params,
            ret,
            outputs: Vec::new(),
            locals: Vec::new(),
            aliases: HashMap::new(),
            views: HashSet::new(),
            changed: Vec::new(),
            ops: Vec::new(),
            count: 0,
            cursor: Vec::new(),
            loops: Vec::new(),
            label: None,
            head: None,
        }
    }

    fn db(&self) -> &'a TypeDb {
        self.program.db
    }

    /// Gives the function its `out` parameters, each held in a local of its own. A local starts
    /// at its type's zero on every call, as an output does.
    fn declare_outputs(&mut self, outputs: &[(String, Ty)]) {
        self.outputs = outputs.to_vec();
        for (name, ty) in outputs {
            self.locals.push((out_local(name), ty.clone()));
        }
    }

    /// If `name` is one of the function's `out` parameters, the local that holds it and its type.
    fn output(&self, name: &str) -> Option<(String, Ty)> {
        self.outputs.iter().find(|(n, _)| n == name).map(|(n, t)| (out_local(n), t.clone()))
    }

    /// Returns from the function: a result node fed the return value, if any, and the current
    /// value of each `out` parameter.
    fn result(&mut self, value: Option<Value>) -> Result<()> {
        let node = self.node("return", NodeKind::Result);
        self.chain(&node, None);
        if let Some(value) = value {
            self.feed(value.source, pin(&node, RETURN_PIN));
        }
        for (name, _) in self.outputs.clone() {
            let held = self.name(&name, Span::default())?;
            self.feed(held.source, pin(&node, &name));
        }
        Ok(())
    }

    fn locals_declared(&self) -> Result<Vec<Local>> {
        self.locals.iter().map(|(n, t)| Ok(Local { name: n.clone(), ty: t.op_type()?, default: None })).collect()
    }

    /// Node ids are global in ops.json, so each carries its graph's name.
    fn id(&self, local: &str) -> String {
        format!("{}/{local}", self.name)
    }

    fn node(&mut self, prefix: &str, kind: NodeKind) -> String {
        self.count += 1;
        let id = self.id(&format!("{prefix}{}", self.count));
        self.ops.push(Op::Node { graph: self.graph.clone(), id: id.clone(), kind });
        id
    }

    fn link(&mut self, from: PinRef, to: PinRef) {
        self.ops.push(Op::Link { from, to });
    }

    fn feed(&mut self, value: Source, to: PinRef) {
        match value {
            Source::Pin(from) => self.link(from, to),
            Source::Literal(value) => self.ops.push(Op::Default { pin: to, value }),
            Source::Nothing => {}
            Source::Node(_) => unreachable!("records are checked before they are fed"),
        }
    }

    /// Runs `node` after what came before. Its exec output is `then`, or it has none.
    fn chain(&mut self, node: &str, then: Option<&str>) {
        if let Some(head) = &mut self.head {
            head.get_or_insert_with(|| node.to_string());
        }
        for from in std::mem::take(&mut self.cursor) {
            self.link(from, pin(node, "execute"));
        }
        self.cursor = then.map(|t| vec![pin(node, t)]).unwrap_or_default();
    }

    fn function_body(&mut self, body: &[Stmt]) -> Result<()> {
        let db = self.program.db;
        changed_in(body, &mut self.changed, &|name| changes_container(db, name));
        let entry = self.id("entry");
        self.ops.push(Op::Node { graph: self.name.clone(), id: entry.clone(), kind: NodeKind::Entry });
        self.cursor = vec![pin(&entry, "then")];
        self.block(body)?;
        // The end of the body gives back the outputs as a return does.
        if !self.outputs.is_empty() && !self.cursor.is_empty() {
            self.result(None)?;
        }
        Ok(())
    }

    fn block(&mut self, body: &[Stmt]) -> Result<()> {
        body.iter().try_for_each(|s| self.stmt(s))
    }

    fn local(&self, name: &str) -> Option<Ty> {
        self.locals.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
    }

    /// Whether the function already holds something of this name: a parameter, an output, a
    /// local, a loop's item, or a name for a call's outputs.
    fn declared(&self, name: &str) -> bool {
        self.params.contains_key(name) || self.output(name).is_some() || self.aliases.contains_key(name) || self.local(name).is_some()
    }

    /// Whether a name is a value the function or the Blueprint holds, a dispatcher included, rather
    /// than a function, or an enum or a class named where a value could be.
    fn is_value(&self, name: &str) -> bool {
        self.declared(name)
            || self.program.members.contains_key(name)
            || self.program.dispatchers.contains_key(name)
            || self.db().property(&self.program.parent, name).is_some()
    }

    /// The loop a break or continue leaves or goes on with: the innermost, or the one its label
    /// names.
    fn target_loop(&mut self, word: &str, label: Option<&str>, span: Span) -> Result<&mut Loop> {
        match label {
            None => self.loops.last_mut().ok_or_else(|| anyhow!("{span}: {word} is only allowed inside a loop")),
            Some(label) => self
                .loops
                .iter_mut()
                .rev()
                .find(|l| l.label.as_deref() == Some(label))
                .ok_or_else(|| anyhow!("{span}: no loop around this {word} is labelled {label}")),
        }
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<()> {
        let span = match stmt {
            Stmt::Let { span, .. }
            | Stmt::Assign { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::While { span, .. }
            | Stmt::Break { span, .. }
            | Stmt::Continue { span, .. }
            | Stmt::Labelled { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Count { span, .. }
            | Stmt::Compound { span, .. }
            | Stmt::If { span, .. }
            | Stmt::Match { span, .. } => *span,
            Stmt::Expr(e) => e.span,
        };
        if self.cursor.is_empty() {
            bail!("{span}: this can never run, since every path before it returns, breaks or continues");
        }
        match stmt {
            Stmt::Let { name, ty, value, span } => {
                if self.params.contains_key(name) || self.output(name).is_some() || self.program.members.contains_key(name) || self.aliases.contains_key(name) {
                    bail!("{span}: {name} is already declared");
                }
                let literal_items = match &value.kind {
                    ExprKind::ArrayLiteral(items) => Some(items.as_slice()),
                    ExprKind::EmptyArray if ty.as_ref().is_some_and(|t| t.name.starts_with("array<")) => Some(&[][..]),
                    _ => None,
                };
                if let Some(items) = literal_items {
                    let declared = match ty {
                        Some(t) => self.program.ty(t)?,
                        None => literal_array_type(items)
                            .ok_or_else(|| anyhow!("{span}: an array literal needs a type unless it holds plain literals: let {name}: array<T> = [...]"))?,
                    };
                    match self.local(name) {
                        Some(existing) if existing != declared => bail!("{span}: {name} is already a {existing}"),
                        Some(_) => {}
                        None => self.locals.push((name.clone(), declared.clone())),
                    }
                    let target = Expr { kind: ExprKind::Name(name.clone()), span: *span };
                    return self.fill_array(&target, &declared, items, *span);
                }
                // A struct, array or map read out of memory, and never changed through this name or
                // the one it is read from, is read where it lies each time it is used, as a
                // Blueprint reads it, rather than copied into a local.
                let viewable = ty.is_none()
                    && read_path(value).is_some_and(|path| path.len() > 1 && path[0] != *name && !self.overlaps_change(&path))
                    && !self.overlaps_change(std::slice::from_ref(name))
                    && self.local(name).is_none();
                if viewable {
                    let read = self.expr(value)?;
                    if matches!(read.ty, Ty::Struct(_) | Ty::Array(_) | Ty::Map(..)) {
                        self.aliases.insert(name.clone(), read);
                        self.views.insert(name.clone());
                        return Ok(());
                    }
                    return self.let_local(name, None, read, *span);
                }
                let value = match ty {
                    Some(t) => {
                        let declared = self.program.ty(t)?;
                        self.arg(value, &declared)?
                    }
                    None => self.expr(value)?,
                };
                self.let_local(name, ty.as_ref(), value, *span)?;
            }
            Stmt::Assign { target, value, span } => self.assign(target, value, *span)?,
            Stmt::Compound { target, op, value, span } => {
                let target = self.settle(target, *span)?;
                let combined = Expr { kind: ExprKind::Binary(*op, Box::new(target.clone()), Box::new(value.clone())), span: *span };
                self.assign(&target, &combined, *span)?;
            }
            Stmt::Return { value, span } => {
                let value = match (value, self.ret.clone()) {
                    (Some(value), Some(ret)) => {
                        let value = self.arg(value, &ret)?;
                        self.check(&value, &ret, *span, "the return value")?;
                        Some(value)
                    }
                    (None, None) => None,
                    (Some(_), None) => bail!("{span}: {} returns nothing", self.name),
                    (None, Some(ret)) => bail!("{span}: {} must return a {ret}", self.name),
                };
                if self.graph == EVENT_GRAPH {
                    // An event returns nothing, and its chain just ends.
                    self.cursor.clear();
                    return Ok(());
                }
                self.result(value)?;
            }
            Stmt::While { cond, body, span } => {
                let before = self.head.replace(None);
                let value = self.condition(cond, *span)?;
                let first = std::mem::replace(&mut self.head, before).flatten();
                let branch = self.node("while", NodeKind::Branch);
                self.chain(&branch, Some("then"));
                self.feed(value, pin(&branch, "Condition"));
                // Each pass starts again from the condition, including any call it made.
                let start = first.unwrap_or_else(|| branch.clone());
                self.loops.push(Loop { label: self.label.take(), ..Loop::default() });
                self.block(body)?;
                let inner = self.loops.pop().expect("the loop pushed above");
                self.cursor.extend(inner.continues);
                for from in std::mem::take(&mut self.cursor) {
                    self.link(from, pin(&start, "execute"));
                }
                self.cursor = vec![pin(&branch, "else")];
                self.cursor.extend(inner.breaks);
            }
            Stmt::Break { label, span } => {
                let from = std::mem::take(&mut self.cursor);
                self.target_loop("break", label.as_deref(), *span)?.breaks.extend(from);
            }
            Stmt::Continue { label, span } => {
                let from = std::mem::take(&mut self.cursor);
                self.target_loop("continue", label.as_deref(), *span)?.continues.extend(from);
            }
            Stmt::Labelled { label, looped, span } => {
                if self.loops.iter().any(|l| l.label.as_deref() == Some(label.as_str())) {
                    bail!("{span}: a loop around this one is labelled {label} already");
                }
                self.label = Some(label.clone());
                self.stmt(looped)?;
            }
            Stmt::For { names, items, body, span } => self.for_loop(names, items, body, *span)?,
            Stmt::Count { name, first, last, body, span } => self.count_loop(name, first, last, body, *span)?,
            Stmt::If { cond, then, otherwise, span } => {
                let value = self.condition(cond, *span)?;
                let branch = self.node("if", NodeKind::Branch);
                self.chain(&branch, Some("then"));
                self.feed(value, pin(&branch, "Condition"));
                self.block(then)?;
                let after_then = std::mem::replace(&mut self.cursor, vec![pin(&branch, "else")]);
                self.block(otherwise)?;
                self.cursor.extend(after_then);
            }
            Stmt::Match { value, arms, otherwise, span } => {
                let values: Vec<&[Expr]> = arms.iter().map(|(values, _)| values.as_slice()).collect();
                let conditions = self.arm_conditions(value, &values, *span)?;
                // An if for each arm, each the one before's else.
                let mut chain = otherwise.clone().unwrap_or_default();
                for (cond, (_, body)) in conditions.into_iter().zip(arms).rev() {
                    chain = vec![Stmt::If { cond, then: body.clone(), otherwise: chain, span: *span }];
                }
                self.block(&chain)?;
            }
            Stmt::Expr(expr) => {
                let (pure, _) = self.call_expr(expr)?;
                if pure {
                    bail!("{}: this call is pure, so on its own it does nothing", expr.span);
                }
            }
        }
        Ok(())
    }

    /// Whether something the function changes lies on `path`, or `path` lies within it. An index
    /// stands for every item.
    fn overlaps_change(&self, path: &[String]) -> bool {
        self.changed.iter().any(|changed| changed.iter().zip(path).all(|(a, b)| a == b || a == "[]" || b == "[]"))
    }

    /// `let name[: ty] = value`, with the value lowered: a local set to it, or for a call's
    /// several outputs, a name for the call.
    fn let_local(&mut self, name: &str, ty: Option<&TypeName>, value: Value, span: Span) -> Result<()> {
        if let Ty::Record(_) = value.ty {
            if ty.is_some() || self.local(name).is_some() {
                bail!("{span}: {name} holds a call's outputs, so it cannot have a type or be a local");
            }
            self.aliases.insert(name.to_string(), value);
            return Ok(());
        }
        let ty = match ty {
            Some(t) => {
                let declared = self.program.ty(t)?;
                self.check(&value, &declared, span, name)?;
                declared
            }
            None => value.ty.clone(),
        };
        if let Ty::Delegate(_) = ty {
            bail!("{span}: {name} would hold a {ty}, and Blueprint has no delegate variables; pass it straight to what takes it");
        }
        // A let that runs again, in a loop, or a second let of the same name and type,
        // reuses the local.
        match self.local(name) {
            Some(existing) if existing != ty => {
                bail!("{span}: {name} is already a {existing}, so it cannot be a {ty} too")
            }
            Some(_) => {}
            None if ty == Ty::Null => bail!("{span}: none needs a type: let {name}: SomeClass = none"),
            None => self.locals.push((name.to_string(), ty)),
        }
        self.set(NodeKind::Set { var: name.to_string(), scope: Scope::Local, class: None }, name, None, value);
        Ok(())
    }

    fn condition(&mut self, cond: &Expr, span: Span) -> Result<Source> {
        let value = self.expr(cond)?;
        if value.ty != Ty::Bool {
            bail!("{span}: the condition is {}, not bool", value.ty);
        }
        Ok(value.source)
    }

    /// Fails unless `value` can go where `to` is expected.
    fn check(&self, value: &Value, to: &Ty, span: Span, what: &str) -> Result<()> {
        if let Ty::Record(_) = value.ty {
            bail!("{span}: {what} is {to}, but the value is {}; pick one by name", value.ty);
        }
        if !self.db().assignable(&value.ty, to) {
            bail!("{span}: {what} is {to}, but the value is {}", value.ty);
        }
        Ok(())
    }

    /// A set node on the chain, fed `value`, and with `target` as the object it writes to.
    fn set(&mut self, kind: NodeKind, pin_name: &str, target: Option<Source>, value: Value) {
        let node = self.node("set", kind);
        self.chain(&node, Some("then"));
        if let Some(target) = target {
            self.feed(target, pin(&node, "self"));
        }
        self.feed(value.source, pin(&node, pin_name));
    }

    /// `target = [a, b, c]`: empties the array, then adds each item in order. There is no node
    /// that makes an array from values of a type known in advance, so this is statements.
    fn fill_array(&mut self, target: &Expr, ty: &Ty, items: &[Expr], span: Span) -> Result<()> {
        if !matches!(ty, Ty::Array(_)) {
            bail!("{span}: an array literal can only fill an array, not a {ty}");
        }
        // A set node cannot leave an array unlinked, so the array is emptied in place.
        let array = self.expr(target)?;
        self.container_call(array, "Clear", &[], span)?;
        for item in items {
            // an int written out stands for a float in an array of floats, as it does elsewhere
            let item = match (&item.kind, ty) {
                (ExprKind::Int(n), Ty::Array(inner)) if **inner == Ty::Float => Expr { kind: ExprKind::Float(*n as f64), span: item.span },
                _ => item.clone(),
            };
            let array = self.expr(target)?;
            self.container_call(array, "Add", std::slice::from_ref(&item), item.span)?;
        }
        Ok(())
    }

    /// Refuses, in a `pure fn`, something that changes what is outside the function.
    fn effect(&self, span: Span, what: impl std::fmt::Display) -> Result<()> {
        if self.checks_effects {
            bail!(
                "{span}: {} is pure, so it runs each time its value is read, and {what}; write `pure mut fn` to let it, or leave out `pure`",
                self.name
            );
        }
        Ok(())
    }

    /// The type of a place the function holds itself: a local, or a field, item or value of a
    /// struct, array or map it holds. None for anything else, such as a member or an object's
    /// property, whose change is seen outside the function.
    fn own_place(&self, e: &Expr) -> Option<Ty> {
        match &e.kind {
            ExprKind::Name(name) => self.output(name).map(|(_, ty)| ty).or_else(|| self.local(name)),
            ExprKind::Member(holder, field) => match self.own_place(holder)? {
                Ty::Struct(path) => self.db().field(&path, field),
                _ => None,
            },
            ExprKind::Index(holder, _) => match self.own_place(holder)? {
                Ty::Array(inner) => Some(*inner),
                Ty::Map(_, value) => Some(*value),
                _ => None,
            },
            _ => None,
        }
    }

    fn assign(&mut self, target: &Expr, value: &Expr, span: Span) -> Result<()> {
        if self.own_place(target).is_none() {
            self.effect(span, "this assigns what is outside it")?;
        }
        match &value.kind {
            ExprKind::ArrayLiteral(items) => {
                let ty = self.expr_type_of_target(target, span)?;
                return self.fill_array(target, &ty, items, span);
            }
            ExprKind::EmptyArray if matches!(target.kind, ExprKind::Name(_)) => {
                let ty = self.expr_type_of_target(target, span)?;
                return self.fill_array(target, &ty, &[], span);
            }
            ExprKind::MapLiteral(entries) if entries.is_empty() => {
                let map = self.expr(target)?;
                if !matches!(map.ty, Ty::Map(..)) {
                    bail!("{span}: {{}} empties a map, and this is a {}", map.ty);
                }
                self.container_call(map, "Clear", &[], span)?;
                return Ok(());
            }
            ExprKind::MapLiteral(_) => bail!("{span}: a map literal can only be a variable's default; add entries with map[key] = value"),
            _ => {}
        }
        match &target.kind {
            ExprKind::Name(name) => {
                // An output is assigned as the local that holds it, whose pin is named after the local.
                let held = self.output(name);
                let (ty, kind) = if let Some((local, t)) = &held {
                    (t.clone(), NodeKind::Set { var: local.clone(), scope: Scope::Local, class: None })
                } else if let Some(t) = self.local(name) {
                    (t, NodeKind::Set { var: name.clone(), scope: Scope::Local, class: None })
                } else if let Some(t) = self.program.members.get(name) {
                    (t.clone(), NodeKind::Set { var: name.clone(), scope: Scope::SelfMember, class: None })
                } else if self.params.contains_key(name) {
                    bail!("{span}: {name} is a parameter and cannot be assigned");
                } else if self.aliases.contains_key(name) {
                    bail!("{span}: {name} is a loop's item or count, and cannot be assigned");
                } else if let Some(p) = self.db().property(&self.program.parent, name) {
                    if p.read_only {
                        bail!("{span}: {name} is read-only");
                    }
                    let class = self.db().property_owner(&self.program.parent, name);
                    (p.ty, NodeKind::Set { var: name.clone(), scope: Scope::Inherited, class })
                } else {
                    bail!("{span}: no variable {name}");
                };
                let value = self.arg(value, &ty)?;
                self.check(&value, &ty, span, name)?;
                let pin_name = held.as_ref().map_or(name.as_str(), |(local, _)| local.as_str());
                self.set(kind, pin_name, None, value);
            }
            ExprKind::Member(object, field) if matches!(object.kind, ExprKind::Name(_)) && matches!(self.expr_type_of_target(object, span), Ok(Ty::Struct(_))) => {
                // A struct variable's field, set in place.
                let Ok(Ty::Struct(path)) = self.expr_type_of_target(object, span) else { unreachable!() };
                if let Some((class, function)) = self.db().native_make(&path) {
                    // The engine makes such a struct with a function of its own, and cannot set one
                    // field of it, so it is made again with that field changed.
                    let make = self.program.library(&class, &function)?;
                    if !make.params.iter().any(|p| p.name == *field) {
                        bail!("{span}: {} has no field {field}", short(&path));
                    }
                    let fields = make
                        .params
                        .iter()
                        .map(|p| {
                            let v = if p.name == *field {
                                value.clone()
                            } else {
                                Expr { kind: ExprKind::Member(object.clone(), p.name.clone()), span }
                            };
                            (p.name.clone(), v)
                        })
                        .collect();
                    let remade = Expr { kind: ExprKind::Make(short(&path).to_string(), fields), span };
                    return self.assign(object, &remade, span);
                }
                let Some(field_ty) = self.db().field(&path, field) else {
                    bail!("{span}: {} has no field {field} that Blueprints can set", short(&path));
                };
                if self.db().field_read_only(&path, field) {
                    bail!("{span}: {}'s {field} is read-only, so Blueprints can read it but not set it", short(&path));
                }
                let value = self.arg(value, &field_ty)?;
                self.check(&value, &field_ty, span, field)?;
                let target = self.expr(object)?;
                let node = self.node("set", NodeKind::SetFields { r#struct: path.clone(), fields: vec![field.clone()] });
                self.chain(&node, Some("then"));
                self.feed(target.source, pin(&node, "StructRef"));
                self.feed(value.source, pin(&node, field));
            }
            ExprKind::Member(target, field) => {
                let object = self.expr(target)?;
                if let (Ty::Struct(path), ExprKind::Member(holder, _)) = (&object.ty, &target.kind) {
                    // A field of an object's struct property is set in place: the property's
                    // getter feeds Set Members by reference, so the struct is not copied out and
                    // written back whole (costly, and unsafe, for a large struct the game owns).
                    let path = path.clone();
                    if self.db().native_make(&path).is_none() && matches!(self.expr_type_of_target(holder, span), Ok(Ty::Object(_))) {
                        let Some(field_ty) = self.db().field(&path, field) else {
                            bail!("{span}: {} has no field {field} that Blueprints can set", short(&path));
                        };
                        if self.db().field_read_only(&path, field) {
                            bail!("{span}: {}'s {field} is read-only, so Blueprints can read it but not set it", short(&path));
                        }
                        let value = self.arg(value, &field_ty)?;
                        self.check(&value, &field_ty, span, field)?;
                        let node = self.node("set", NodeKind::SetFields { r#struct: path.clone(), fields: vec![field.clone()] });
                        self.chain(&node, Some("then"));
                        self.feed(object.source, pin(&node, "StructRef"));
                        self.feed(value.source, pin(&node, field));
                        return Ok(());
                    }
                }
                if let Ty::Struct(_) = &object.ty {
                    // A field of a struct that is not a variable of its own, such as an object's
                    // property or an array's item: the struct is copied into a local, the field is
                    // set there, and the struct is put back where it came from.
                    self.count += 1;
                    let local = format!("{RESERVED_PREFIX}Struct{}", self.count);
                    self.locals.push((local.clone(), object.ty.clone()));
                    self.set(NodeKind::Set { var: local.clone(), scope: Scope::Local, class: None }, &local, None, object);
                    let copy = Expr { kind: ExprKind::Name(local), span };
                    let member = Expr { kind: ExprKind::Member(Box::new(copy.clone()), field.clone()), span };
                    self.assign(&member, value, span)?;
                    return self.assign(target, &copy, span);
                }
                let Ty::Object(class) = &object.ty else {
                    bail!("{span}: {} has no members", object.ty);
                };
                let Some(p) = self.db().property(class, field) else {
                    bail!("{span}: {}", no_member(self.db(), class, field));
                };
                if p.read_only {
                    bail!("{span}: {field} is read-only");
                }
                let owner = self.db().property_owner(class, field);
                let value = self.arg(value, &p.ty)?;
                self.check(&value, &p.ty, span, field)?;
                let kind = NodeKind::Set { var: field.clone(), scope: Scope::External, class: owner };
                self.set(kind, field, Some(object.source), value);
            }
            ExprKind::Index(array, index) => {
                let array = self.expr(array)?;
                if let Ty::Map(..) = array.ty {
                    self.container_call(array, "Add", &[(**index).clone(), value.clone()], span)?;
                    return Ok(());
                }
                let Ty::Array(inner) = &array.ty else { bail!("{span}: {} is not an array", array.ty) };
                let inner = (**inner).clone();
                let index = self.expr(index)?;
                if index.ty != Ty::Int {
                    bail!("{span}: an index is an int, not {}", index.ty);
                }
                let value = self.expr(value)?;
                self.check(&value, &inner, span, "the item")?;
                let node = self.node("call", NodeKind::Call { function: format!("{ARRAY}:Array_Set") });
                self.chain(&node, Some("then"));
                // The array first, so the node knows its item type before the item is linked.
                self.feed(array.source, pin(&node, "TargetArray"));
                self.feed(index.source, pin(&node, "Index"));
                self.feed(value.source, pin(&node, "Item"));
            }
            _ => bail!("{span}: only a variable, a member or an item can be assigned"),
        }
        Ok(())
    }

    /// The type of something that can be assigned, without lowering it.
    fn expr_type_of_target(&mut self, target: &Expr, span: Span) -> Result<Ty> {
        match &target.kind {
            ExprKind::Name(name) => {
                if let Some((_, t)) = self.output(name) {
                    return Ok(t);
                }
                if let Some(t) = self.local(name) {
                    return Ok(t);
                }
                if let Some(t) = self.program.members.get(name) {
                    return Ok(t.clone());
                }
                if let Some(p) = self.db().property(&self.program.parent, name) {
                    return Ok(p.ty);
                }
                bail!("{span}: no variable {name}")
            }
            _ => bail!("{span}: an array literal can fill a variable, not this"),
        }
    }

    /// A method of a container or a soft reference, from the table of them.
    fn method(&self, on: On, name: &str, args: usize, span: Span) -> Result<&'static builtins::Method> {
        builtins::method(on, name, args)
            .ok_or_else(|| anyhow!("{span}: {} has {}, not {name} with {args} arguments", on.a(), builtins::methods_of(on)))
    }

    /// The object if it is of `class`, and none otherwise. Pure.
    fn cast(&mut self, value: Value, class: &str) -> Value {
        let node = self.node("cast", NodeKind::Cast { class: class.to_string() });
        self.feed(value.source, pin(&node, "Object"));
        Value { source: Source::Pin(pin(&node, "Result")), ty: Ty::Object(class.to_string()) }
    }

    /// The object as `interface` if it implements it, and none otherwise. Pure.
    fn interface_cast(&mut self, value: Value, interface: &str) -> Value {
        let node = self.node("cast", NodeKind::Cast { class: interface.to_string() });
        self.feed(value.source, pin(&node, "Object"));
        Value { source: Source::Pin(pin(&node, "Result")), ty: Ty::Interface(interface.to_string()) }
    }

    /// The class if it is `class` or a subclass, and none otherwise. Pure.
    fn class_cast(&mut self, value: Value, class: &str) -> Value {
        let node = self.node("cast", NodeKind::ClassCast { class: class.to_string() });
        self.feed(value.source, pin(&node, "Class"));
        Value { source: Source::Pin(pin(&node, "Result")), ty: Ty::Class(class.to_string()) }
    }

    /// A function of a container's library called on it, by the function's name without its
    /// library's prefix: `Items.RemoveItem(4)` calls `Array_RemoveItem`. Its types that the
    /// engine ties to the container are the container's.
    fn container_call(&mut self, container: Value, name: &str, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        let Some((function, receiver)) = self.db().container_function(&container.ty, name).map_err(|e| anyhow!("{span}: {e}"))? else {
            let on = match container.ty {
                Ty::Array(_) => "an array",
                Ty::Set(_) => "a set",
                _ => "a map",
            };
            bail!("{span}: {on} has no method {name}; it has {}", self.db().container_methods(&container.ty).join(", "));
        };
        let mut values = self.args_for(&function, args, span)?;
        let mut whole = function;
        whole.params.insert(0, crate::types::Param { name: receiver, ty: container.ty.clone(), optional: false, by_ref: false, wired: true });
        values.insert(0, container);
        self.call_node(&whole, None, values, false)
    }

    /// One output of a call that gives several, as a value.
    fn record_output(record: &Value, name: &str) -> Value {
        let (Source::Node(node), Ty::Record(fields)) = (&record.source, &record.ty) else { unreachable!("a call with outputs gives a record") };
        let ty = fields.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone()).expect("the output the caller names");
        Value { source: Source::Pin(pin(node, name)), ty }
    }

    /// `map[key]`: the value the map holds for the key, or the value type's zero when it holds none.
    fn map_find(&mut self, map: Value, key: Value, span: Span) -> Result<Value> {
        let Ty::Map(key_ty, value_ty) = &map.ty else { unreachable!() };
        self.check(&key, key_ty, span, "the key")?;
        let value_ty = (**value_ty).clone();
        let node = self.node("pure", NodeKind::Call { function: format!("{MAP}:Map_Find") });
        self.feed(map.source, pin(&node, "TargetMap"));
        self.feed(key.source, pin(&node, "Key"));
        Ok(Value { source: Source::Pin(pin(&node, "Value")), ty: value_ty })
    }

    /// A soft reference's methods: `load()` loads what it refers to and waits for it, and
    /// `valid()` says whether it refers to anything.
    fn soft_method(&mut self, soft: Value, name: &str, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        let method = self.method(On::Soft, name, args.len(), span)?;
        let of_class = matches!(soft.ty, Ty::SoftClass(_));
        let function = if of_class { method.class_function.unwrap_or(method.function) } else { method.function };
        let pure = method.pure;
        // Loading gives an object or a class of any class, which is cast back to the reference's.
        let (param, ret) = match (name, of_class) {
            ("load", false) => ("Asset", Ty::Object(OBJECT.into())),
            ("load", true) => ("AssetClass", Ty::Class(OBJECT.into())),
            (_, false) => ("SoftObjectReference", Ty::Bool),
            (_, true) => ("SoftClassReference", Ty::Bool),
        };
        let library = self.program.library(SYSTEM, function)?;
        let reference = self.wired_arg(&library, param, soft.clone());
        let node = self.node(if pure { "pure" } else { "call" }, NodeKind::Call { function: format!("{SYSTEM}:{function}") });
        if !pure {
            self.chain(&node, Some("then"));
        }
        self.feed(reference.source, pin(&node, param));
        let loaded = Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: ret };
        Ok((pure, Some(match &soft.ty {
            Ty::SoftObject(class) if name == "load" && class != OBJECT => self.cast(loaded, class),
            Ty::SoftClass(class) if name == "load" && class != OBJECT => self.class_cast(loaded, class),
            _ => loaded,
        })))
    }

    /// `for item in array { }` walks the array by index, and `for item, i in array { }` names the
    /// index too; `for key, value in map { }` walks the map's keys, taken once before the first
    /// pass, and finds each one's value.
    fn for_loop(&mut self, names: &[String], items: &Expr, body: &[Stmt], span: Span) -> Result<()> {
        for name in names {
            if self.declared(name) || self.program.members.contains_key(name) {
                bail!("{span}: {name} is already declared");
            }
        }
        let collection = self.expr(items)?;
        let (array, map) = match (&collection.ty, names.len()) {
            // The parser takes one or two names, and an array's second is its index.
            (Ty::Array(_), _) => (collection.clone(), None),
            (Ty::Set(_), 1) => {
                let (_, items) = self.container_call(collection.clone(), "ToArray", &[], span)?;
                (Self::record_output(&items.expect("ToArray gives its output"), "Result"), None)
            }
            (Ty::Set(_), _) => bail!("{span}: for over a set names one item: for item in set"),
            (Ty::Map(..), 2) => {
                let (_, keys) = self.container_call(collection.clone(), "Keys", &[], span)?;
                (Self::record_output(&keys.expect("Keys gives its output"), "Keys"), Some(collection.clone()))
            }
            (Ty::Map(..), _) => bail!("{span}: for over a map names a key and a value: for key, value in map"),
            (ty, _) => bail!("{span}: for walks an array, a set or a map, not {ty}"),
        };
        let Ty::Array(item_ty) = &array.ty else { unreachable!() };
        let item_ty = (**item_ty).clone();

        self.count += 1;
        let index = format!("{RESERVED_PREFIX}Index{}", self.count);
        self.locals.push((index.clone(), Ty::Int));
        let zero = Value { source: Source::Literal("0".into()), ty: Ty::Int };
        self.set(NodeKind::Set { var: index.clone(), scope: Scope::Local, class: None }, &index, None, zero);

        let length = self.node("pure", NodeKind::Call { function: format!("{ARRAY}:Array_Length") });
        self.feed(array.source.clone(), pin(&length, "TargetArray"));
        let current = self.name(&index, span)?;
        let more = self.pure_library(
            MATH,
            "Less_IntInt",
            vec![("A", current), ("B", Value { source: Source::Pin(pin(&length, "ReturnValue")), ty: Ty::Int })],
            Ty::Bool,
        )?;
        let branch = self.node("for", NodeKind::Branch);
        self.chain(&branch, Some("then"));
        self.feed(more.source, pin(&branch, "Condition"));

        // The item is read where it is used, at the index of the pass that uses it.
        let get = self.node("item", NodeKind::ArrayGet);
        self.feed(array.source, pin(&get, "Array"));
        let current = self.name(&index, span)?;
        self.feed(current.source, pin(&get, "Dimension 1"));
        let item = Value { source: Source::Pin(pin(&get, "Output")), ty: item_ty };
        self.aliases.insert(names[0].clone(), item.clone());
        if let Some(map) = map {
            let value = self.map_find(map, item, span)?;
            self.aliases.insert(names[1].clone(), value);
        } else if let Some(position) = names.get(1) {
            // An array's second name is the item's index, read-only as a count's number is.
            let current = self.name(&index, span)?;
            self.aliases.insert(position.clone(), current);
        }
        self.loops.push(Loop { label: self.label.take(), ..Loop::default() });
        self.block(body)?;
        let inner = self.loops.pop().expect("the loop pushed above");
        for name in names {
            self.aliases.remove(name);
        }

        // A continue goes on to the next item as the end of the body does.
        self.cursor.extend(inner.continues);
        if !self.cursor.is_empty() {
            let current = self.name(&index, span)?;
            let next = self.pure_library(MATH, "Add_IntInt", vec![("A", current), ("B", Value { source: Source::Literal("1".into()), ty: Ty::Int })], Ty::Int)?;
            self.set(NodeKind::Set { var: index.clone(), scope: Scope::Local, class: None }, &index, None, next);
            for from in std::mem::take(&mut self.cursor) {
                self.link(from, pin(&branch, "execute"));
            }
        }
        self.cursor = vec![pin(&branch, "else")];
        self.cursor.extend(inner.breaks);
        Ok(())
    }

    /// The target of a compound assignment, which is read and then written, with what picks it out
    /// computed once: an index not written out, and an object a call gives, are kept in locals of
    /// their own first. An array or struct the target is in is written in place, so it stays as
    /// written.
    fn settle(&mut self, target: &Expr, span: Span) -> Result<Expr> {
        let kind = match &target.kind {
            ExprKind::Member(object, field) => ExprKind::Member(Box::new(self.settle_holder(object, span)?), field.clone()),
            ExprKind::Index(items, key) => {
                let key = match key.kind {
                    ExprKind::Int(_) | ExprKind::Str(_) | ExprKind::Float(_) | ExprKind::Bool(_) => (**key).clone(),
                    _ => self.hoist(key, span)?,
                };
                ExprKind::Index(Box::new(self.settle_holder(items, span)?), Box::new(key))
            }
            _ => return Ok(target.clone()),
        };
        Ok(Expr { kind, span: target.span })
    }

    /// What holds a compound assignment's target: a path, settled in turn, or an object a call
    /// gives, kept in a local.
    fn settle_holder(&mut self, holder: &Expr, span: Span) -> Result<Expr> {
        match holder.kind {
            ExprKind::Name(_) | ExprKind::Member(..) | ExprKind::Index(..) => self.settle(holder, span),
            _ => self.hoist(holder, span),
        }
    }

    /// A value computed now into a local of its own, and the name that reads it.
    fn hoist(&mut self, value: &Expr, span: Span) -> Result<Expr> {
        self.count += 1;
        let name = format!("{RESERVED_PREFIX}Target{}", self.count);
        let read = self.expr(value)?;
        self.let_local(&name, None, read, span)?;
        Ok(Expr { kind: ExprKind::Name(name), span: value.span })
    }

    /// `for i in first..last`, as Blueprint's For Loop counts: from first to last, both included,
    /// with `i` read from a local of the loop's own. `first` is computed once, before the first
    /// pass, and `last` again before every pass, as the macro reads its LastIndex pin.
    fn count_loop(&mut self, name: &str, first: &Expr, last: &Expr, body: &[Stmt], span: Span) -> Result<()> {
        if self.declared(name) || self.program.members.contains_key(name) {
            bail!("{span}: {name} is already declared");
        }
        let first_value = self.arg(first, &Ty::Int)?;
        self.check(&first_value, &Ty::Int, first.span, "a count's first number")?;

        self.count += 1;
        let index = format!("{RESERVED_PREFIX}Index{}", self.count);
        self.locals.push((index.clone(), Ty::Int));
        self.set(NodeKind::Set { var: index.clone(), scope: Scope::Local, class: None }, &index, None, first_value);

        // Each pass starts again from `last`, including any call it makes, as a while's
        // condition does.
        let before = self.head.replace(None);
        let last_value = self.arg(last, &Ty::Int)?;
        self.check(&last_value, &Ty::Int, last.span, "a count's last number")?;
        let current = self.name(&index, span)?;
        let more = self.pure_library(MATH, "LessEqual_IntInt", vec![("A", current), ("B", last_value)], Ty::Bool)?;
        let first_node = std::mem::replace(&mut self.head, before).flatten();
        let branch = self.node("for", NodeKind::Branch);
        self.chain(&branch, Some("then"));
        self.feed(more.source, pin(&branch, "Condition"));
        let start = first_node.unwrap_or_else(|| branch.clone());

        let current = self.name(&index, span)?;
        self.aliases.insert(name.to_string(), current);
        self.loops.push(Loop { label: self.label.take(), ..Loop::default() });
        self.block(body)?;
        let inner = self.loops.pop().expect("the loop pushed above");
        self.aliases.remove(name);

        // A continue goes on to the next number as the end of the body does.
        self.cursor.extend(inner.continues);
        if !self.cursor.is_empty() {
            let current = self.name(&index, span)?;
            let next = self.pure_library(MATH, "Add_IntInt", vec![("A", current), ("B", Value { source: Source::Literal("1".into()), ty: Ty::Int })], Ty::Int)?;
            self.set(NodeKind::Set { var: index.clone(), scope: Scope::Local, class: None }, &index, None, next);
            for from in std::mem::take(&mut self.cursor) {
                self.link(from, pin(&start, "execute"));
            }
        }
        self.cursor = vec![pin(&branch, "else")];
        self.cursor.extend(inner.breaks);
        Ok(())
    }

    /// Lowers a call, returning whether it was pure and what it returned.
    fn call_expr(&mut self, expr: &Expr) -> Result<(bool, Option<Value>)> {
        match &expr.kind {
            ExprKind::Call(name, args, quoted) => self.named_call(name, args, *quoted, expr.span),
            ExprKind::Method(object, name, args) => self.method_call(object, name, args, expr.span),
            _ => bail!("{}: only a call can stand alone", expr.span),
        }
    }

    /// A call by name. One in backticks is never a built-in function, but the Blueprint's own.
    fn named_call(&mut self, name: &str, args: &[Expr], quoted: bool, span: Span) -> Result<(bool, Option<Value>)> {
        if let Some(ty @ Ty::Delegate(_)) = self.params.get(name).cloned() {
            bail!("{span}: {name} is a {ty}, and Blueprint cannot call a delegate; pass it on to what takes one");
        }
        if let Some(value) = self.enum_from(name, args, span)? {
            return Ok((true, Some(value)));
        }
        // A struct's name with nothing passed makes the struct with every field at its zero, as
        // `Vector(X: 1)` makes one with every other field at its zero.
        if args.is_empty() && !quoted && !self.program.functions.contains_key(name) && builtins::function(name).is_none() {
            if self.program.structure(name).map_err(|e| anyhow!("{span}: {e}"))?.is_some() {
                let made = Expr { kind: ExprKind::Make(name.to_string(), Vec::new()), span };
                return Ok((true, Some(self.expr(&made)?)));
            }
        }
        if !quoted {
            if matches!(name, "create_widget" | "construct") {
                self.effect(span, format!("it calls {name}, which makes an object"))?;
            }
            if let Some(value) = self.builtin(name, args, span)? {
                return Ok((!matches!(name, "create_widget" | "construct"), Some(value)));
            }
        }
        // A function of an interface this Blueprint implements, called on this object as on any
        // other, even where this Blueprint's own handler implements it.
        let interface = match &self.program.class {
            Some(class) if !self.db().has_function(&self.program.parent, name) => {
                self.program.interface_with(class, name).map_err(|e| anyhow!("{span}: {e}"))?
            }
            _ => None,
        };
        if let Some(interface) = interface {
            let target = self.name("self", span)?;
            return self.message(&interface, name, target, args, span);
        }
        if let Some(function) = self.program.functions.get(name) {
            let required = function.defaults.iter().filter(|d| d.is_none()).count();
            if args.len() < required || args.len() > function.params.len() {
                let names: Vec<String> = function.params.iter().map(|(n, t, _)| format!("{n}: {t}")).collect();
                if args.len() > function.params.len() && !function.outputs.is_empty() {
                    // An argument too many, where an output may have been taken for an input.
                    let outputs: Vec<&str> = function.outputs.iter().map(|(n, _)| n.as_str()).collect();
                    bail!(
                        "{span}: {name} takes ({}); {} {} its outputs, which a call does not pass but reads, as let r = {name}(...) then r.{}",
                        names.join(", "),
                        outputs.join(" and "),
                        if outputs.len() == 1 { "is" } else { "are" },
                        outputs[0]
                    );
                }
                bail!("{span}: {name} takes ({}), and {required} of them are required", names.join(", "));
            }
            let params = function.params.clone();
            let defaults = function.defaults.clone();
            let ret = function.ret.clone();
            let outputs = function.outputs.clone();
            let pure = function.pure;
            if !pure {
                self.effect(span, format!("it calls {name}, which is not pure"))?;
            }
            let mut values = Vec::new();
            for (arg, (param, ty, by_ref)) in args.iter().zip(&params) {
                let value = if *by_ref { self.ref_arg(arg, param)? } else { self.arg(arg, ty)? };
                self.check(&value, ty, arg.span, param)?;
                values.push((param.clone(), value.source));
            }
            // A parameter left out takes its default, written on the call as the editor would.
            for ((param, ..), default) in params.iter().zip(&defaults).skip(args.len()) {
                let default = default.clone().expect("every parameter left out has a default");
                values.push((param.clone(), Source::Literal(default)));
            }
            // A pure function is a node off the chain, run each time its value is read.
            let node = self.node(if pure { "pure" } else { "call" }, NodeKind::Call { function: format!("self:{name}") });
            if !pure {
                self.chain(&node, Some("then"));
            }
            for (param, value) in values {
                self.feed(value, pin(&node, &param));
            }
            // With outputs, the call gives a record of them and the return value, read by name, as
            // an engine function's outputs are given.
            if !outputs.is_empty() {
                let mut fields = outputs;
                fields.extend(ret.map(|ty| (RETURN_PIN.to_string(), ty)));
                return Ok((pure, Some(Value { source: Source::Node(node), ty: Ty::Record(fields) })));
            }
            let result = ret.map(|ty| Value { source: Source::Pin(pin(&node, RETURN_PIN)), ty });
            return Ok((pure, result));
        }
        // A function of the parent class, called on this object.
        match self.program.function(&self.program.parent, name).map_err(|e| anyhow!("{span}: {e}"))? {
            Some(function) => self.call(&function, None, args, span),
            None => bail!("{span}: no function {name}"),
        }
    }

    fn method_call(&mut self, object: &Expr, name: &str, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        if matches!(&object.kind, ExprKind::Name(n) if n == "super") {
            return self.super_call(name, args, span);
        }
        let binds = matches!(name, "bind" | "unbind" | "unbind_all");
        if let Some(params) = self.dispatcher(object) {
            if binds {
                let ExprKind::Name(dispatcher) = &object.kind else { unreachable!() };
                return self.dispatcher_method(None, dispatcher, name, args, span);
            }
            return self.broadcast(object, params, name, args, span);
        }
        if let (true, ExprKind::Member(owner, dispatcher)) = (binds, &object.kind) {
            return self.dispatcher_method(Some(owner), dispatcher, name, args, span);
        }
        if let Some(class) = self.class_name(object)? {
            let Some(function) = self.program.function(&class, name).map_err(|e| anyhow!("{span}: {e}"))? else {
                bail!("{span}: {} has no function {name}", short(&class));
            };
            if !function.is_static {
                bail!("{span}: {name} is not static, so it needs an object of {}", short(&class));
            }
            return self.call(&function, None, args, span);
        }
        let target = self.expr(object)?;
        if matches!(target.ty, Ty::Array(_) | Ty::Map(..) | Ty::Set(_)) && self.checks_effects && self.own_place(object).is_none() {
            if self.db().container_function(&target.ty, name).ok().flatten().is_some_and(|(f, _)| !f.pure) {
                self.effect(span, format!("{name} changes what is outside it"))?;
            }
        }
        match target.ty {
            Ty::Array(_) | Ty::Map(..) | Ty::Set(_) => return self.container_call(target, name, args, span),
            Ty::SoftObject(_) | Ty::SoftClass(_) => return self.soft_method(target, name, args, span),
            _ => {}
        }
        if let Ty::Interface(interface) = &target.ty {
            let interface = interface.clone();
            return self.message(&interface, name, target, args, span);
        }
        let Ty::Object(class) = &target.ty else {
            bail!("{span}: {} has no functions", target.ty);
        };
        let Some(function) = self.program.function(class, name).map_err(|e| anyhow!("{span}: {e}"))? else {
            // A function of an interface the object's class implements.
            if let Some(interface) = self.program.interface_with(class, name).map_err(|e| anyhow!("{span}: {e}"))? {
                return self.message(&interface, name, target, args, span);
            }
            bail!("{span}: {} has no function {name}", short(class));
        };
        self.call(&function, Some(target.source), args, span)
    }

    /// `object.Function(args)`, for a function of an interface: a message, which does nothing to an
    /// object that does not implement the interface, and gives each output its zero.
    fn message(&mut self, interface: &str, name: &str, target: Value, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        let Some(function) = self.program.function(interface, name).map_err(|e| anyhow!("{span}: {e}"))? else {
            bail!("{span}: {} has no function {name}", short(interface));
        };
        self.effect(span, format!("it calls {name} through {}, which is never pure", short(interface)))?;
        let values = self.args_for(&function, args, span)?;
        let target = self.as_object(target)?;
        self.call_node(&function, Some(target.source), values, true)
    }

    /// An interface's value as the object it is, for a pin that takes an object. Anything else is
    /// itself.
    fn as_object(&mut self, value: Value) -> Result<Value> {
        let Ty::Interface(_) = value.ty else { return Ok(value) };
        self.pure_library(SYSTEM, "Conv_InterfaceToObject", vec![("Interface", value)], Ty::Object(OBJECT.into()))
    }

    /// `Dispatcher.bind(Function)`: the dispatcher's broadcasts call Function from then on, which
    /// takes what the dispatcher passes. `unbind(Function)` stops that, and `unbind_all()` stops
    /// it for everything bound. The dispatcher is this Blueprint's, or with `owner`, an object's.
    fn dispatcher_method(&mut self, owner: Option<&Expr>, dispatcher: &str, method: &str, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        self.effect(span, format!("it changes what {dispatcher} calls"))?;
        let own = |graph: &Self| {
            graph.program.dispatchers.get(dispatcher).map(|params| {
                params.iter().map(|(name, ty)| Slot { name: name.clone(), ty: ty.clone(), passing: Passing::of_blueprint(ty, false), floats: Floats::Double }).collect::<Vec<_>>()
            })
        };
        // This Blueprint's own dispatcher is on this object, whose class may not be built yet.
        let is_self = owner.is_none_or(|o| matches!(&o.kind, ExprKind::Name(n) if n == "self"));
        let (class, object, signature) = match (owner, own(self)) {
            (_, Some(signature)) if is_self => (None, None, signature),
            (None, _) => unreachable!("a dispatcher named alone is this Blueprint's"),
            (Some(owner), own_signature) => {
                let object = self.expr(owner)?;
                let Ty::Object(class) = &object.ty else { bail!("{span}: {} has no dispatchers", object.ty) };
                let signature = match own_signature.filter(|_| self.program.class.as_ref() == Some(class)) {
                    Some(signature) => signature,
                    None => match self.db().dispatcher_signature(class, dispatcher).map_err(|e| anyhow!("{span}: {e}"))? {
                        Some(signature) => signature,
                        None => bail!("{span}: {} has no dispatcher {dispatcher}", short(class)),
                    },
                };
                (Some(class.clone()), Some(object.source), signature)
            }
        };
        let dispatcher = dispatcher.to_string();
        let (node, delegate) = match (method, args) {
            ("bind" | "unbind", [function]) => {
                let delegate = self.delegate(function, &signature, &dispatcher)?;
                let kind = if method == "bind" {
                    NodeKind::AddDelegate { class, dispatcher }
                } else {
                    NodeKind::RemoveDelegate { class, dispatcher }
                };
                (self.node(method, kind), Some(delegate))
            }
            ("bind" | "unbind", _) => bail!("{span}: {method}() takes a function, as in {dispatcher}.{method}(OnDone)"),
            (_, []) => (self.node("unbind_all", NodeKind::ClearDelegate { class, dispatcher }), None),
            _ => bail!("{span}: unbind_all() takes nothing"),
        };
        self.chain(&node, Some("then"));
        if let Some(object) = object {
            self.feed(object, pin(&node, "self"));
        }
        if let Some(delegate) = delegate {
            self.feed(delegate, pin(&node, "Delegate"));
        }
        Ok((false, None))
    }

    /// A delegate where one of `signature` is expected, which `to` names: to the function `e`
    /// names, as Blueprint's Create Event makes one, or a delegate `e` holds already, such as a
    /// parameter. The function takes exactly what the signature passes, as Blueprint checks, and
    /// is one Blueprint can bind.
    fn delegate(&mut self, e: &Expr, signature: &[Slot], to: &str) -> Result<Source> {
        let (function, object, shape, own) = match self.callee(e)? {
            Callee::Function { name, object, shape, own } => (name, object, shape, own),
            Callee::Value(value) => {
                let Ty::Delegate(path) = &value.ty else {
                    bail!("{}: this is a {}, where a function is expected, named by itself as in bind(OnDone)", e.span, value.ty);
                };
                let held = self.db().signature(path).map_err(|err| anyhow!("{}: {err}", e.span))?;
                if !same_shape(&held, signature, true) {
                    bail!("{}: this {} is {}, and a function bound to {to} must be {}", e.span, value.ty, describe(&held), describe(signature));
                }
                return Ok(value.source);
            }
        };
        if !same_shape(&shape, signature, false) {
            bail!(
                "{}: {function} is {}, and a function bound to {to} must be {}, with each parameter of the same type and passed the same way",
                e.span,
                describe(&shape),
                describe(signature)
            );
        }
        if own {
            self.narrow(&function, signature, to, e.span)?;
        } else if let Some((f, s)) = shape.iter().zip(signature).find(|(f, s)| f.floats != s.floats) {
            // Blueprint would bind it through a proxy, which no unbind matches, and for another
            // object's function, or for some of the engine's dispatchers, would crash compiling it.
            let bits = |floats: Floats| if floats == Floats::Double { "64-bit" } else { "32-bit" };
            bail!(
                "{}: {function} takes {} as a {} float, and {to} passes a {} one; bind a function of this Blueprint's own instead, whose floats loom makes as wide as {to}'s",
                e.span,
                f.name,
                bits(f.floats),
                bits(s.floats)
            );
        }
        let node = self.node("delegate", NodeKind::Delegate { function });
        if let Some(object) = object {
            self.feed(object, pin(&node, "self"));
        }
        Ok(Source::Pin(pin(&node, "OutputDelegate")))
    }

    /// Makes this Blueprint's `function`, bound to `to`, take each float as wide as `signature`
    /// passes it. A function bound to two delegates that pass one of its floats at different
    /// widths is refused, since it can't take both.
    fn narrow(&self, function: &str, signature: &[Slot], to: &str, span: Span) -> Result<()> {
        if let Some(slot) = signature.iter().find(|s| s.floats == Floats::SingleInside) {
            bail!("{span}: {to} passes {} as a {} of 32-bit floats, and loom cannot yet bind a function to it", slot.name, slot.ty);
        }
        if let Some(slot) = signature.iter().find(|s| s.floats == Floats::Single && s.passing == Passing::Ref) {
            bail!("{span}: {to} passes {} by reference as a 32-bit float, and loom cannot yet bind a function to it", slot.name);
        }
        let widths: Vec<Floats> = signature.iter().map(|s| s.floats).collect();
        let mut narrowed = self.program.narrowed.borrow_mut();
        match narrowed.get(function) {
            Some(earlier) if earlier.widths != widths => {
                let (i, _) = widths.iter().zip(&earlier.widths).enumerate().find(|(_, (a, b))| a != b).expect("the widths differ somewhere");
                let bits = |floats: Floats| if floats == Floats::Single { "32-bit" } else { "64-bit" };
                bail!(
                    "{span}: {function} is bound to {to}, which passes {} as a {} float, and at {} to {}, which passes a {} one; bind a function to each",
                    signature[i].name,
                    bits(widths[i]),
                    earlier.span,
                    earlier.to,
                    bits(earlier.widths[i])
                );
            }
            Some(_) => {}
            None => {
                narrowed.insert(function.to_string(), Narrowed { widths, span, to: to.to_string() });
            }
        }
        Ok(())
    }

    /// What `e` names where a delegate is expected: a function Blueprint can bind, this
    /// Blueprint's, its parent's or an object's, or else a value.
    fn callee(&mut self, e: &Expr) -> Result<Callee> {
        let bindable = |function: &Function| -> Result<Vec<Slot>> {
            let name = &function.name;
            if function.pure {
                bail!("{}: {name} is pure, and Blueprint binds only a function with exec pins", e.span);
            }
            if function.latent {
                bail!("{}: {name} waits, and Blueprint binds no function that waits", e.span);
            }
            if function.deprecated {
                bail!("{}: {name} is deprecated, and Blueprint binds no deprecated function", e.span);
            }
            if function.is_static {
                bail!("{}: {name} is static, so it belongs to no object for a delegate to call it on", e.span);
            }
            Ok(function.shape.clone())
        };
        match &e.kind {
            ExprKind::Name(name) if name != "self" && !self.is_value(name) => {
                if let Some(declared) = self.program.functions.get(name) {
                    if self.program.class.is_none() {
                        bail!("{}: {name} is a function library's, which is static, so it belongs to no object for a delegate to call it on", e.span);
                    }
                    if declared.pure {
                        bail!("{}: {name} is pure, and Blueprint binds only a function with exec pins", e.span);
                    }
                    let mut shape: Vec<Slot> = declared
                        .params
                        .iter()
                        .map(|(n, t, by_ref)| Slot { name: n.clone(), ty: t.clone(), passing: Passing::of_blueprint(t, *by_ref), floats: Floats::Double })
                        .collect();
                    // Its outputs follow its inputs, as Unreal orders a Blueprint function's parameters.
                    shape.extend(declared.outputs.iter().map(|(n, t)| Slot { name: n.clone(), ty: t.clone(), passing: Passing::Out, floats: Floats::Double }));
                    shape.extend(declared.ret.iter().map(|t| Slot { name: RETURN_PIN.to_string(), ty: t.clone(), passing: Passing::Return, floats: Floats::Double }));
                    return Ok(Callee::Function { name: name.clone(), object: None, shape, own: true });
                }
                if let Some(function) = self.program.function(&self.program.parent, name).map_err(|err| anyhow!("{}: {err}", e.span))? {
                    let shape = bindable(&function)?;
                    return Ok(Callee::Function { name: name.clone(), object: None, shape, own: false });
                }
            }
            ExprKind::Member(holder, name) if self.enum_name(holder)?.is_none() => {
                if let Some(class) = self.class_name(holder)? {
                    if self.db().has_function(&class, name) {
                        bail!("{}: {name} is static, so it belongs to no object for a delegate to call it on", e.span);
                    }
                    return Ok(Callee::Value(self.expr(e)?));
                }
                let object = self.expr(holder)?;
                let Ty::Object(class) = &object.ty else { return Ok(Callee::Value(self.member(object, name, e.span)?)) };
                if self.db().property(class, name).is_some() {
                    return Ok(Callee::Value(self.member(object, name, e.span)?));
                }
                // This Blueprint's own function, on another object of its class, or on this one.
                if self.program.class.as_ref() == Some(class) && self.program.functions.contains_key(name) {
                    let Callee::Function { shape, .. } = self.callee(&Expr { kind: ExprKind::Name(name.clone()), span: e.span })? else {
                        unreachable!("a function of this Blueprint is one")
                    };
                    return Ok(Callee::Function { name: name.clone(), object: Some(object.source), shape, own: true });
                }
                let Some(function) = self.program.function(class, name).map_err(|err| anyhow!("{}: {err}", e.span))? else {
                    return Ok(Callee::Value(self.member(object, name, e.span)?));
                };
                let shape = bindable(&function)?;
                return Ok(Callee::Function { name: name.clone(), object: Some(object.source), shape, own: false });
            }
            _ => {}
        }
        Ok(Callee::Value(self.expr(e)?))
    }

    /// If `e` names one of this Blueprint's dispatchers, its parameters.
    fn dispatcher(&self, e: &Expr) -> Option<Vec<(String, Ty)>> {
        let ExprKind::Name(name) = &e.kind else { return None };
        if self.declared(name) {
            return None;
        }
        self.program.dispatchers.get(name).cloned()
    }

    /// `Dispatcher.broadcast(args)`: calls everything bound to it.
    fn broadcast(&mut self, object: &Expr, params: Vec<(String, Ty)>, method: &str, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        let ExprKind::Name(dispatcher) = &object.kind else { unreachable!() };
        self.effect(span, format!("it broadcasts {dispatcher}"))?;
        if method != "broadcast" {
            bail!("{span}: {dispatcher} is a dispatcher, which has broadcast(), bind(), unbind() and unbind_all()");
        }
        if args.len() != params.len() {
            let names: Vec<String> = params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
            bail!("{span}: {dispatcher} passes ({}), not {} arguments", names.join(", "), args.len());
        }
        let mut values = Vec::new();
        for (arg, (param, ty)) in args.iter().zip(&params) {
            let value = self.arg(arg, ty)?;
            self.check(&value, ty, arg.span, param)?;
            values.push((param.clone(), value.source));
        }
        let node = self.node("broadcast", NodeKind::Broadcast { dispatcher: dispatcher.clone() });
        self.chain(&node, Some("then"));
        for (param, value) in values {
            self.feed(value, pin(&node, &param));
        }
        Ok((false, None))
    }

    /// `a == b` or `a != b` on enum values. The node learns the enum from a linked pin, so one
    /// side must be more than a value written out.
    fn enum_equal(&mut self, a: Value, b: Value, negate: bool, span: Span) -> Result<Value> {
        let (first, second) = match (&a.source, &b.source) {
            (Source::Pin(_), _) => (a, b),
            (_, Source::Pin(_)) => (b, a),
            _ => bail!("{span}: comparing two enum values written out always gives the same answer"),
        };
        let Ty::Enum(path) = &first.ty else {
            bail!("{span}: {} is not an enum", first.ty);
        };
        let path = path.clone();
        // A value written out comes from a literal node rather than the pin's own default. The
        // comparison copies the other input's type, and an array's item is read by reference,
        // which leaves a pin that no longer takes a value written out.
        let second = match second.source {
            Source::Literal(value) => {
                let literal = self.node("pure", NodeKind::EnumLiteral { r#enum: path.clone() });
                self.feed(Source::Literal(value), pin(&literal, "Enum"));
                Source::Pin(pin(&literal, "ReturnValue"))
            }
            other => other,
        };
        let node = self.node("pure", NodeKind::EnumEqual { r#enum: path });
        self.feed(first.source, pin(&node, "A"));
        self.feed(second, pin(&node, "B"));
        let equal = Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: Ty::Bool };
        if !negate {
            return Ok(equal);
        }
        self.pure_library(MATH, "Not_PreBool", vec![("A", equal)], Ty::Bool)
    }

    /// If `e` names an enum rather than a variable, the enum.
    fn enum_name(&self, e: &Expr) -> Result<Option<String>> {
        let ExprKind::Name(name) = &e.kind else { return Ok(None) };
        if self.is_value(name) {
            return Ok(None);
        }
        self.program.enumeration(name).map_err(|err| anyhow!("{}: {err}", e.span))
    }

    /// If `e` names a class rather than a variable, the class.
    fn class_name(&self, e: &Expr) -> Result<Option<String>> {
        let ExprKind::Name(name) = &e.kind else { return Ok(None) };
        if self.is_value(name) || self.program.dispatchers.contains_key(name) {
            return Ok(None);
        }
        self.program.class(name).map_err(|err| anyhow!("{}: {err}", e.span))
    }

    fn call(&mut self, function: &Function, target: Option<Source>, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        if !function.pure {
            self.effect(span, format!("it calls {}, which is not pure", function.name))?;
        }
        if function.latent && self.graph != EVENT_GRAPH {
            self.waits = self.handler;
            bail!("{span}: {} waits, so it can only be called in an event's `on` handler or an `event`", function.name);
        }
        let values = self.args_for(function, args, span)?;
        self.call_values(function, target, values)
    }

    /// The arguments of a call, lowered and checked against the function's parameters.
    fn args_for(&mut self, function: &Function, args: &[Expr], span: Span) -> Result<Vec<Value>> {
        let required = function.params.iter().filter(|p| !p.optional).count();
        if args.len() < required || args.len() > function.params.len() {
            let names: Vec<String> = function.params.iter().map(|p| format!("{}: {}", p.name, p.ty)).collect();
            bail!("{span}: {} takes ({}), and {} of them are required", function.name, names.join(", "), required);
        }
        let mut values = Vec::new();
        for (arg, param) in args.iter().zip(&function.params) {
            let value = if param.by_ref { self.ref_arg(arg, &param.name)? } else { self.arg(arg, &param.ty)? };
            self.check(&value, &param.ty, arg.span, &param.name)?;
            values.push(value);
        }
        Ok(values)
    }

    /// `super.Name(args)`, in the handler or function that overrides Name: the parent's own
    /// version of it, as the editor's "Add call to parent function" makes.
    fn super_call(&mut self, name: &str, args: &[Expr], span: Span) -> Result<(bool, Option<Value>)> {
        self.effect(span, format!("it calls the parent's {name}"))?;
        if self.name != name && self.name != event_function(name) {
            bail!("{span}: super.{name} calls the parent's {name}, so it goes in the `on {name}` or `fn {name}` that overrides it");
        }
        let parent = self.program.parent.clone();
        let Some(function) = self.db().function_in(&parent, name, &["not_callable"]).map_err(|e| anyhow!("{span}: {e}"))? else {
            bail!("{span}: {} has no {name} to call", short(&parent));
        };
        let values = self.args_for(&function, args, span)?;
        let node = self.node("super", NodeKind::CallParent { function: format!("{}:{}", function.owner, function.name) });
        self.chain(&node, Some("then"));
        for (value, param) in values.into_iter().zip(&function.params) {
            self.feed(value.source, pin(&node, &param.name));
        }
        let ret = function.ret.clone().map(|ty| Value { source: Source::Pin(pin(&node, RETURN_PIN)), ty });
        Ok((false, ret))
    }

    /// A call whose arguments are lowered and checked already, one per parameter from the first.
    fn call_values(&mut self, function: &Function, target: Option<Source>, args: Vec<Value>) -> Result<(bool, Option<Value>)> {
        self.call_node(function, target, args, false)
    }

    /// A call node, or with `message`, an interface's message node, which is never pure.
    fn call_node(&mut self, function: &Function, target: Option<Source>, args: Vec<Value>, message: bool) -> Result<(bool, Option<Value>)> {
        let mut values = Vec::new();
        let mut output_class = None;
        for (value, param) in args.into_iter().zip(&function.params) {
            if function.determines_output_type.as_deref() == Some(param.name.as_str()) {
                if let Ty::Class(class) = &value.ty {
                    output_class = Some(class.clone());
                }
            }
            let value = self.wired_arg(function, &param.name, value);
            values.push((param.name.clone(), value.source));
        }
        let pure = function.pure && !message;
        let reference = format!("{}:{}", function.owner, function.name);
        let node = if message {
            self.node("message", NodeKind::Message { function: reference })
        } else {
            self.node(if pure { "pure" } else { "call" }, NodeKind::Call { function: reference })
        };
        if !pure {
            self.chain(&node, Some("then"));
        }
        if let Some(target) = target {
            self.feed(target, pin(&node, "self"));
        }
        for (param, value) in values {
            self.feed(value, pin(&node, &param));
        }
        // The node retypes its output to the class it was given, as the editor does.
        let ret = match (function.ret.clone(), output_class.clone()) {
            (Some(Ty::Object(_)), Some(class)) => Some(Ty::Object(class)),
            (Some(Ty::Class(_)), Some(class)) => Some(Ty::Class(class)),
            (Some(Ty::Array(inner)), Some(class)) if matches!(*inner, Ty::Object(_)) => {
                Some(Ty::Array(Box::new(Ty::Object(class))))
            }
            (ret, _) => ret,
        };
        if !function.outputs.is_empty() {
            let mut fields = function.outputs.clone();
            if let Some(class) = &output_class {
                for (_, ty) in &mut fields {
                    match ty {
                        Ty::Object(_) => *ty = Ty::Object(class.clone()),
                        Ty::Array(inner) if matches!(**inner, Ty::Object(_)) => *ty = Ty::Array(Box::new(Ty::Object(class.clone()))),
                        _ => {}
                    }
                }
            }
            if let Some(ret) = ret {
                fields.push(("ReturnValue".to_string(), ret));
            }
            return Ok((pure, Some(Value { source: Source::Node(node), ty: Ty::Record(fields) })));
        }
        let result = ret.map(|ty| Value { source: Source::Pin(pin(&node, "ReturnValue")), ty });
        Ok((pure, result))
    }

    /// `if cond { a } else { b }`: a branch that sets a local of its own on each side, which is
    /// then read, so only the side chosen is computed. With `to`, each side is an argument of that
    /// type, and otherwise they agree on one.
    fn choose(&mut self, cond: &Expr, a: &Expr, b: &Expr, span: Span, to: Option<&Ty>) -> Result<Value> {
        // The value chosen is held in a local, which a delegate cannot be.
        let no_delegate = |ty: &Ty| match ty {
            Ty::Delegate(_) => bail!("{span}: an if or a match cannot choose a {ty}, since Blueprint has no delegate variables; use an if statement"),
            _ => Ok(()),
        };
        if let Some(to) = to {
            no_delegate(to)?;
        }
        let value = self.condition(cond, span)?;
        let branch = self.node("if", NodeKind::Branch);
        self.chain(&branch, Some("then"));
        self.feed(value, pin(&branch, "Condition"));
        let mut then = match to {
            Some(to) => self.arg(a, to)?,
            None => self.expr(a)?,
        };
        let after_then = std::mem::replace(&mut self.cursor, vec![pin(&branch, "else")]);
        let otherwise = self.arg(b, to.unwrap_or(&then.ty))?;
        let ty = match to {
            Some(to) => to.clone(),
            None => match (&then.ty, &otherwise.ty, &a.kind) {
                (Ty::Int, Ty::Float, ExprKind::Int(n)) => {
                    then = Value { source: Source::Literal(format!("{n}.0")), ty: Ty::Float };
                    Ty::Float
                }
                (Ty::Null, Ty::Null, _) => bail!("{span}: both sides are none, which needs a type: let x: SomeClass = ..."),
                (Ty::Null, other, _) => other.clone(),
                (one, other, _) if self.db().assignable(other, one) => one.clone(),
                (one, other, _) if self.db().assignable(one, other) => other.clone(),
                (one, other, _) => bail!("{span}: one side is {one} and the other {other}; both sides give the same type"),
            },
        };
        no_delegate(&ty)?;
        self.check(&then, &ty, a.span, "this side")?;
        self.check(&otherwise, &ty, b.span, "this side")?;

        self.count += 1;
        let local = format!("{RESERVED_PREFIX}If{}", self.count);
        self.locals.push((local.clone(), ty.clone()));
        let set = || NodeKind::Set { var: local.clone(), scope: Scope::Local, class: None };
        let after_else = std::mem::replace(&mut self.cursor, after_then);
        self.set(set(), &local, None, then);
        let done = std::mem::replace(&mut self.cursor, after_else);
        self.set(set(), &local, None, otherwise);
        self.cursor.extend(done);
        let node = self.node("get", NodeKind::Get { var: local.clone(), scope: Scope::Local, class: None });
        Ok(Value { source: Source::Pin(pin(&node, &local)), ty })
    }

    /// `match value { A { a } else { b } }`, as the `if` expression `if value == A { a } else { b }`.
    fn match_expr(&mut self, value: &Expr, arms: &[(Vec<Expr>, Expr)], otherwise: &Expr, span: Span, to: Option<&Ty>) -> Result<Value> {
        let values: Vec<&[Expr]> = arms.iter().map(|(values, _)| values.as_slice()).collect();
        let conditions = self.arm_conditions(value, &values, span)?;
        let mut chain = otherwise.clone();
        for (cond, (_, arm)) in conditions.into_iter().zip(arms).rev() {
            chain = Expr { kind: ExprKind::If(Box::new(cond), Box::new(arm.clone()), Box::new(chain)), span };
        }
        match to {
            Some(to) => self.arg(&chain, to),
            None => self.expr(&chain),
        }
    }

    /// A match's value, put in a local of its own on the chain, and for each arm, the condition it
    /// is chosen on: the local equal to one of the values the arm lists.
    fn arm_conditions(&mut self, value: &Expr, arms: &[&[Expr]], span: Span) -> Result<Vec<Expr>> {
        if arms.is_empty() {
            bail!("{span}: a match needs an arm that lists values");
        }
        let matched = self.expr(value)?;
        if !matches!(matched.ty, Ty::Int | Ty::Str | Ty::Name | Ty::Enum(_)) {
            bail!("{span}: a match compares an int, a string, a name or an enum value, not {}", matched.ty);
        }
        self.count += 1;
        let local = format!("{RESERVED_PREFIX}Match{}", self.count);
        self.locals.push((local.clone(), matched.ty.clone()));
        self.set(NodeKind::Set { var: local.clone(), scope: Scope::Local, class: None }, &local, None, matched);

        let mut listed = HashSet::new();
        let mut conditions = Vec::new();
        for values in arms {
            let mut cond: Option<Expr> = None;
            for v in *values {
                let written = self.written_value(v)?;
                if !listed.insert(written.clone()) {
                    bail!("{}: {written} is listed by an earlier arm", v.span);
                }
                let name = Expr { kind: ExprKind::Name(local.clone()), span: v.span };
                let equal = Expr { kind: ExprKind::Binary(BinaryOp::Eq, Box::new(name), Box::new(v.clone())), span: v.span };
                cond = Some(match cond {
                    Some(before) => Expr { kind: ExprKind::Binary(BinaryOp::Or, Box::new(before), Box::new(equal)), span: v.span },
                    None => equal,
                });
            }
            conditions.push(cond.expect("the parser gives an arm a value"));
        }
        Ok(conditions)
    }

    /// A value an arm lists, as it is written, which must be a number, a string or `Enum.Value`.
    fn written_value(&self, e: &Expr) -> Result<String> {
        Ok(match &e.kind {
            ExprKind::Int(n) => n.to_string(),
            ExprKind::Unary(UnaryOp::Neg, inner) if matches!(inner.kind, ExprKind::Int(_)) => {
                let ExprKind::Int(n) = inner.kind else { unreachable!() };
                format!("-{n}")
            }
            ExprKind::Str(s) => format!("{s:?}"),
            ExprKind::Member(object, value) if self.enum_name(object)?.is_some() => {
                let ExprKind::Name(name) = &object.kind else { unreachable!() };
                format!("{name}.{value}")
            }
            _ => bail!("{}: an arm lists values written out: a number, a string, or Enum.Value", e.span),
        })
    }

    /// An argument, where an int literal may stand for a float and a string literal for a key.
    fn arg(&mut self, arg: &Expr, to: &Ty) -> Result<Value> {
        if let ExprKind::If(cond, a, b) = &arg.kind {
            return self.choose(cond, a, b, arg.span, Some(to));
        }
        if let ExprKind::Match(value, arms, otherwise) = &arg.kind {
            return self.match_expr(value, arms, otherwise, arg.span, Some(to));
        }
        // A function named where a delegate is expected is a delegate to it.
        if let Ty::Delegate(path) = to {
            let signature = self.db().signature(path).map_err(|e| anyhow!("{}: {e}", arg.span))?;
            let source = self.delegate(arg, &signature, delegate_name(path))?;
            return Ok(Value { source, ty: to.clone() });
        }
        if let (ExprKind::Int(n), Ty::Float) = (&arg.kind, to) {
            return Ok(Value { source: Source::Literal(format!("{n}.0")), ty: Ty::Float });
        }
        // A string written out stands for a name where a name is expected.
        if let (ExprKind::Str(text), Ty::Name) = (&arg.kind, to) {
            return Ok(Value { source: Source::Literal(text.clone()), ty: Ty::Name });
        }
        // An asset's path stands for the asset, or a soft reference to it, as the pin's asset
        // picker would give it. A class is checked when the types know it.
        if let (ExprKind::Path(path), Ty::SoftObject(_) | Ty::SoftClass(_) | Ty::Object(_) | Ty::Class(_)) = (&arg.kind, to) {
            if let Ty::Class(base) | Ty::SoftClass(base) = to {
                if self.db().class_in(path, true, &self.program.scope()).ok().flatten().is_some() && !self.db().is_subclass(path, base) {
                    bail!("{}: {} is not a {}", arg.span, short(path), short(base));
                }
            }
            // A class is its own type, so a call whose result it determines, such as
            // GetActorOfClass, gives that class.
            let ty = if let Ty::Class(_) = to { Ty::Class(path.clone()) } else { to.clone() };
            return Ok(Value { source: Source::Literal(path.clone()), ty });
        }
        // A key is written by its name, such as "Left" or "PageUp", which is how its pin holds it.
        if let (ExprKind::Str(name), Ty::Struct(path)) = (&arg.kind, to) {
            if path == KEY {
                return Ok(Value { source: Source::Literal(name.clone()), ty: to.clone() });
            }
        }
        // An array written out is filled into a local of its own, which the call then reads.
        if let (ExprKind::ArrayLiteral(items), Ty::Array(_)) = (&arg.kind, to) {
            self.count += 1;
            let local = format!("{RESERVED_PREFIX}Array{}", self.count);
            self.locals.push((local.clone(), to.clone()));
            let target = Expr { kind: ExprKind::Name(local), span: arg.span };
            self.fill_array(&target, to, items, arg.span)?;
            return self.expr(&target);
        }
        // So is a map, which a local of its own starts out as an empty one.
        if let (ExprKind::MapLiteral(entries), Ty::Map(..)) = (&arg.kind, to) {
            self.count += 1;
            let local = format!("{RESERVED_PREFIX}Map{}", self.count);
            self.locals.push((local.clone(), to.clone()));
            let target = Expr { kind: ExprKind::Name(local), span: arg.span };
            for (key, value) in entries {
                let map = self.expr(&target)?;
                self.container_call(map, "Add", &[key.clone(), value.clone()], key.span)?;
            }
            return self.expr(&target);
        }
        self.expr(arg)
    }

    /// An argument for a by-reference parameter: a variable, which the function may change.
    fn ref_arg(&mut self, arg: &Expr, param: &str) -> Result<Value> {
        let ExprKind::Name(name) = &arg.kind else {
            bail!("{}: {param} is passed by reference, so its argument must be a variable", arg.span);
        };
        if self.views.contains(name) {
            bail!("{}: {name} reads a value where it lies, and {param} may change it; give it a type, let {name}: T = ..., to copy it", arg.span);
        }
        self.expr(arg)
    }

    fn pure_library(&mut self, class: &str, name: &str, args: Vec<(&str, Value)>, ret: Ty) -> Result<Value> {
        self.pure_call(self.program.library(class, name)?, args, ret)
    }

    /// A value for a pin that takes nothing written out on it. One that is written out goes into a
    /// local first, as the editor asks, and the local is wired to the pin.
    fn wire(&mut self, value: Value) -> Value {
        let Source::Literal(_) = value.source else { return value };
        self.count += 1;
        let local = format!("{RESERVED_PREFIX}Value{}", self.count);
        self.locals.push((local.clone(), value.ty.clone()));
        let ty = value.ty.clone();
        self.set(NodeKind::Set { var: local.clone(), scope: Scope::Local, class: None }, &local, None, value);
        let node = self.node("get", NodeKind::Get { var: local.clone(), scope: Scope::Local, class: None });
        Value { source: Source::Pin(pin(&node, &local)), ty }
    }

    /// A value for one of a function's inputs, through a local when the input is wired. Every call
    /// to an engine function puts its arguments through this before its node is made, so that the
    /// local is set before the call runs.
    fn wired_arg(&mut self, function: &Function, param: &str, value: Value) -> Value {
        if function.params.iter().any(|p| p.name == param && p.wired) { self.wire(value) } else { value }
    }

    fn pure_call(&mut self, function: Function, args: Vec<(&str, Value)>, ret: Ty) -> Result<Value> {
        let mut values = Vec::new();
        for (param, value) in args {
            if !function.params.iter().any(|p| p.name == param) {
                bail!("{}:{} has no parameter {param}", function.owner, function.name);
            }
            values.push((param, self.wired_arg(&function, param, value).source));
        }
        let node = self.node("pure", NodeKind::Call { function: format!("{}:{}", function.owner, function.name) });
        for (param, value) in values {
            self.feed(value, pin(&node, param));
        }
        Ok(Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: ret })
    }

    /// The functions the language gives, from the table of them. Conversions are spelled as calls
    /// to the type's name.
    /// `Enum(number)`: the value of the enum that the number stands for, named by the enum or
    /// not, as the editor's Byte to Enum node gives it when told not to check.
    fn enum_from(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Option<Value>> {
        if self.declared(name) || self.program.functions.contains_key(name) {
            return Ok(None);
        }
        let Some(path) = self.program.enumeration(name).map_err(|err| anyhow!("{span}: {err}"))? else {
            return Ok(None);
        };
        let [arg] = args else { bail!("{span}: {name}() takes one number, the value it stands for") };
        let value = self.arg(arg, &Ty::Int)?;
        let byte = match value.ty {
            Ty::Byte => value.source,
            Ty::Int => self.pure_library(MATH, "Conv_IntToByte", vec![("InInt", value)], Ty::Byte)?.source,
            other => bail!("{}: {name}() takes a number, not {other}", arg.span),
        };
        let node = self.node("enum", NodeKind::ByteToEnum { r#enum: path.clone() });
        self.feed(byte, pin(&node, "Byte"));
        Ok(Some(Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: Ty::Enum(path) }))
    }

    fn builtin(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Option<Value>> {
        if builtins::function(name).is_none() {
            return Ok(None);
        }
        if name == "create_widget" {
            let [arg] = args else { bail!("{span}: create_widget() takes a widget Blueprint's class") };
            let value = self.expr(arg)?;
            let Ty::Class(class) = &value.ty else { bail!("{span}: create_widget() takes a class, not {}", value.ty) };
            if !self.db().is_subclass(class, USER_WIDGET) {
                bail!("{span}: {} is not a widget", short(class));
            }
            let node = self.node("widget", NodeKind::CreateWidget { class: class.clone() });
            self.chain(&node, Some("then"));
            return Ok(Some(Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: Ty::Object(class.clone()) }));
        }
        if name == "include" {
            bail!("{span}: include() can only be a variable's default");
        }
        if name == "construct" {
            let [class, outer] = args else { bail!("{span}: construct() takes a class and the object to hold the new one") };
            let class = self.expr(class)?;
            let Ty::Class(class) = &class.ty else { bail!("{span}: construct() takes a class, not {}", class.ty) };
            if self.db().is_subclass(class, ACTOR) {
                bail!("{span}: {} is an actor, which is spawned rather than constructed", short(class));
            }
            if self.db().is_subclass(class, USER_WIDGET) {
                bail!("{span}: {} is a widget Blueprint; make it with create_widget()", short(class));
            }
            let outer = self.expr(outer)?;
            self.check(&outer, &Ty::Object(OBJECT.into()), span, "outer")?;
            let node = self.node("construct", NodeKind::Construct { class: class.clone() });
            // the node's outer pin is named self
            self.feed(outer.source, pin(&node, "self"));
            self.chain(&node, Some("then"));
            return Ok(Some(Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: Ty::Object(class.clone()) }));
        }
        if name == "cast" {
            let [value, class] = args else { bail!("{span}: cast() takes an object and a class, as in cast(asset, Blueprint)") };
            let value = self.expr(value)?;
            let class = self.expr(class)?;
            let Ty::Class(class) = class.ty else { bail!("{span}: cast() casts to a class, not {}", class.ty) };
            return Ok(Some(match value.ty {
                Ty::Object(_) | Ty::Interface(_) if self.db().is_interface(&class) => {
                    let object = self.as_object(value)?;
                    self.interface_cast(object, &class)
                }
                Ty::Object(_) | Ty::Interface(_) => {
                    let object = self.as_object(value)?;
                    self.cast(object, &class)
                }
                Ty::Class(_) => self.class_cast(value, &class),
                other => bail!("{span}: cast() takes an object or a class, not {other}"),
            }));
        }
        if name == "implements" {
            let [value, interface] = args else {
                bail!("{span}: implements() takes an object and an interface, as in implements(thing, SomeInterface)")
            };
            let value = self.expr(value)?;
            if !matches!(value.ty, Ty::Object(_) | Ty::Interface(_)) {
                bail!("{span}: implements() takes an object, not {}", value.ty);
            }
            let value = self.as_object(value)?;
            let interface = self.expr(interface)?;
            match &interface.ty {
                Ty::Class(class) if self.db().is_interface(class) => {}
                Ty::Class(class) => bail!("{span}: {} is not an interface", short(class)),
                other => bail!("{span}: implements() takes an interface, not {other}"),
            }
            let function = self.program.library(SYSTEM, "DoesImplementInterface")?;
            return self.pure_call(function, vec![("TestObject", value), ("Interface", interface)], Ty::Bool).map(Some);
        }
        if name == "subsystem" {
            let [class] = args else { bail!("{span}: subsystem() takes a subsystem's class") };
            let class = self.expr(class)?;
            let Ty::Class(class) = class.ty else { bail!("{span}: subsystem() takes a class, not {}", class.ty) };
            if !SUBSYSTEMS.iter().any(|base| self.db().is_subclass(&class, base)) {
                bail!("{span}: {} is not an engine or editor subsystem", short(&class));
            }
            let node = self.node("subsystem", NodeKind::Subsystem { class: class.clone() });
            return Ok(Some(Value { source: Source::Pin(pin(&node, "ReturnValue")), ty: Ty::Object(class) }));
        }
        let [arg] = args else { bail!("{span}: {name}() takes one argument") };
        let value = self.expr(arg)?;
        let Some(conversion) = builtins::conversion(name, &value.ty) else {
            bail!("{span}: {name}() cannot convert {}", value.ty)
        };
        let ret = builtins::plain_type(name).expect("a conversion is to a plain type");
        // Each step takes the last one's value, typed as that step gives it.
        let mut value = value;
        for step in conversion.steps {
            let allowed: &[&str] = if step.internal { &["internal"] } else { &[] };
            let function = self.program.library_in(step.class, step.function, allowed)?;
            let step_ret = function.ret.clone().unwrap_or_else(|| ret.clone());
            value = self.pure_call(function, vec![(step.param, value)], step_ret)?;
        }
        Ok(Some(Value { source: value.source, ty: ret }))
    }

    fn expr(&mut self, e: &Expr) -> Result<Value> {
        let literal = |source: String, ty: Ty| Ok(Value { source: Source::Literal(source), ty });
        match &e.kind {
            ExprKind::Int(n) => literal(n.to_string(), Ty::Int),
            ExprKind::Float(n) => literal(n.to_string(), Ty::Float),
            ExprKind::Bool(b) => literal(b.to_string(), Ty::Bool),
            ExprKind::Str(s) => literal(s.clone(), Ty::Str),
            ExprKind::Null | ExprKind::EmptyArray => Ok(Value { source: Source::Nothing, ty: Ty::Null }),
            ExprKind::ArrayLiteral(_) => bail!("{}: an array literal can only be assigned to an array variable", e.span),
            ExprKind::MapLiteral(_) => bail!("{}: a map literal can only be a variable's default", e.span),
            ExprKind::Path(path) => bail!("{}: {path} is a path, which can only be given to a call that takes that asset or class, or a soft reference to it", e.span),
            ExprKind::If(cond, a, b) => self.choose(cond, a, b, e.span, None),
            ExprKind::Match(value, arms, otherwise) => self.match_expr(value, arms, otherwise, e.span, None),
            ExprKind::Make(name, fields) => {
                let Some(path) = self.program.structure(name).map_err(|err| anyhow!("{}: {err}", e.span))? else {
                    bail!("{}: no struct {name}", e.span);
                };
                if let Some((class, function)) = self.db().native_make(&path) {
                    // Such a struct is made by its function, whose parameters are the fields.
                    let function = self.program.library(&class, &function)?;
                    let mut args = Vec::new();
                    for param in &function.params {
                        match fields.iter().find(|(f, _)| *f == param.name) {
                            Some((_, value)) => args.push(value.clone()),
                            None if param.optional => break,
                            None => bail!("{}: {name} needs {}", e.span, param.name),
                        }
                    }
                    if let Some((field, _)) = fields.iter().find(|(f, _)| !function.params.iter().any(|p| p.name == *f)) {
                        bail!("{}: {name} has no field {field}", e.span);
                    }
                    return match self.call(&function, None, &args, e.span)? {
                        (_, Some(value)) => Ok(value),
                        _ => bail!("{}: {} returns nothing", e.span, function.name),
                    };
                }
                let mut values = Vec::new();
                for (field, value) in fields {
                    let Some(ty) = self.db().field(&path, field) else {
                        bail!("{}: {name} has no field {field} that Blueprints can set", value.span);
                    };
                    if self.db().field_read_only(&path, field) {
                        bail!("{}: {name}'s {field} is read-only, so Blueprints can read it but not make a {name} with it", value.span);
                    }
                    let v = self.arg(value, &ty)?;
                    self.check(&v, &ty, value.span, field)?;
                    values.push((field.clone(), v.source));
                }
                let node = self.node("make", NodeKind::Make { r#struct: path.clone() });
                for (field, value) in values {
                    self.feed(value, pin(&node, &field));
                }
                Ok(Value { source: Source::Pin(pin(&node, short(&path))), ty: Ty::Struct(path) })
            }
            ExprKind::Index(array, index) => {
                let array = self.expr(array)?;
                if let Ty::Map(key, _) = &array.ty {
                    let key = (**key).clone();
                    let key = self.arg(index, &key)?;
                    return self.map_find(array, key, e.span);
                }
                let Ty::Array(inner) = &array.ty else { bail!("{}: {} is not an array", e.span, array.ty) };
                let inner = (**inner).clone();
                let index = self.expr(index)?;
                if index.ty != Ty::Int {
                    bail!("{}: an index is an int, not {}", e.span, index.ty);
                }
                let node = self.node("item", NodeKind::ArrayGet);
                self.feed(array.source, pin(&node, "Array"));
                self.feed(index.source, pin(&node, "Dimension 1"));
                Ok(Value { source: Source::Pin(pin(&node, "Output")), ty: inner })
            }
            ExprKind::Name(name) => self.name(name, e.span),
            ExprKind::Member(object, field) => {
                if let Some(path) = self.enum_name(object)? {
                    let value = enum_value(self.db(), &path, field, e.span)?;
                    return Ok(Value { source: Source::Literal(value), ty: Ty::Enum(path) });
                }
                if self.class_name(object)?.is_some() {
                    bail!("{}: a class has no variables; call its functions", e.span);
                }
                let object = self.expr(object)?;
                self.member(object, field, e.span)
            }
            ExprKind::Call(..) | ExprKind::Method(..) => match self.call_expr(e)? {
                (_, Some(value)) => Ok(value),
                (_, None) => bail!("{}: this returns nothing, so it can only be called as a statement", e.span),
            },
            ExprKind::Unary(op, operand) => {
                let value = self.expr(operand)?;
                let Some(unary) = builtins::unary(*op, &value.ty) else {
                    match op {
                        UnaryOp::Not => bail!("{}: ! needs a bool, not {}", e.span, value.ty),
                        UnaryOp::Neg => bail!("{}: - needs a number, not {}", e.span, value.ty),
                    }
                };
                let args = if unary.zero_first {
                    vec![("A", Value { source: Source::Literal("0".into()), ty: value.ty.clone() }), ("B", value.clone())]
                } else {
                    vec![("A", value.clone())]
                };
                self.pure_library(MATH, unary.function, args, value.ty)
            }
            ExprKind::Binary(op, left, right) => {
                let before = self.cursor.clone();
                let mut a = self.expr(left)?;
                let mut b = self.expr(right)?;
                // A string written out stands for a name where it is compared with one.
                if a.ty == Ty::Name && matches!(right.kind, ExprKind::Str(_)) {
                    b.ty = Ty::Name;
                } else if b.ty == Ty::Name && matches!(left.kind, ExprKind::Str(_)) {
                    a.ty = Ty::Name;
                }
                // A whole number written out stands for a float beside one, as it does for a
                // float argument.
                if let (Ty::Float, ExprKind::Int(n)) = (&a.ty, &right.kind) {
                    b = Value { source: Source::Literal(format!("{n}.0")), ty: Ty::Float };
                } else if let (ExprKind::Int(n), Ty::Float) = (&left.kind, &b.ty) {
                    a = Value { source: Source::Literal(format!("{n}.0")), ty: Ty::Float };
                }
                // An interface's value compares as the object it is.
                let a = self.as_object(a)?;
                let b = self.as_object(b)?;
                if matches!(op, BinaryOp::And | BinaryOp::Or) && self.cursor != before {
                    // Pure nodes evaluate both sides, so && and || cannot short-circuit a call
                    // that does something, or an `if`, which is a branch. A pure call does nothing,
                    // so either side may make one.
                    bail!("{}: {} cannot contain a call that does something or an `if`; split it into ifs", e.span, op.symbol());
                }
                // Two objects compare whatever their classes, and either may be none.
                let object = |t: &Ty| matches!(t, Ty::Object(_) | Ty::Null);
                let on = if object(&a.ty) && object(&b.ty) {
                    if a.ty == Ty::Null && b.ty == Ty::Null {
                        bail!("{}: comparing none with none always gives the same answer", e.span);
                    }
                    Ty::Object(OBJECT.into())
                } else if a.ty != b.ty {
                    bail!("{}: {} {} {} mixes types; convert one side, as with float(), string() or name()", e.span, a.ty, op.symbol(), b.ty);
                } else {
                    a.ty.clone()
                };
                let operator = builtins::operator(*op, &on).ok_or_else(|| anyhow!("{}: {} is not defined for {}", e.span, op.symbol(), a.ty))?;
                match &operator.lowering {
                    Lowering::EnumEqual => self.enum_equal(a, b, *op == BinaryOp::Ne, e.span),
                    Lowering::Library { class, function, returns } => {
                        let ret = returns.ty().expect("an operator gives a plain type");
                        self.pure_library(class, function, vec![("A", a), ("B", b)], ret)
                    }
                }
            }
        }
    }

    /// `object.field`, of a value already lowered: an output of a call's record, a struct's field,
    /// or an object's property.
    fn member(&mut self, object: Value, field: &str, span: Span) -> Result<Value> {
        if let (Ty::Record(fields), Source::Node(node)) = (&object.ty, &object.source) {
            let Some((_, ty)) = fields.iter().find(|(n, _)| n == field) else {
                bail!("{span}: {} has no {field}", object.ty);
            };
            return Ok(Value { source: Source::Pin(pin(node, field)), ty: ty.clone() });
        }
        if let Ty::Struct(path) = object.ty.clone() {
            let path = &path;
            if let Some((class, function)) = self.db().native_break(path) {
                // Such a struct is broken by its function, which has an output per field.
                let function = self.program.library(&class, &function)?;
                let Some((_, Some(record))) = self.call_values(&function, None, vec![object.clone()]).ok() else {
                    bail!("{span}: {} returns nothing", function.name);
                };
                let (Ty::Record(outputs), Source::Node(node)) = (&record.ty, &record.source) else {
                    bail!("{span}: {} has no outputs", function.name);
                };
                let Some((_, ty)) = outputs.iter().find(|(n, _)| n == field) else {
                    bail!("{span}: {} has no {field}", short(path));
                };
                return Ok(Value { source: Source::Pin(pin(node, field)), ty: ty.clone() });
            }
            let Some(ty) = self.db().field(path, field) else {
                bail!("{span}: {} has no field {field} that Blueprints can read", short(path));
            };
            let node = self.node("break", NodeKind::Break { r#struct: path.clone() });
            self.feed(object.source, pin(&node, short(path)));
            return Ok(Value { source: Source::Pin(pin(&node, field)), ty });
        }
        let Ty::Object(class) = &object.ty else {
            bail!("{span}: {} has no members", object.ty);
        };
        let Some(p) = self.db().property(class, field) else {
            // A function named where a value is read, rather than where a delegate is expected.
            if self.program.function(class, field).ok().flatten().is_some() {
                bail!("{span}: {field} is a function of {}; call it, or pass it where a delegate is expected", short(class));
            }
            bail!("{span}: {}", no_member(self.db(), class, field));
        };
        let owner = self.db().property_owner(class, field);
        let node = self.node("get", NodeKind::Get { var: field.to_string(), scope: Scope::External, class: owner });
        self.feed(object.source, pin(&node, "self"));
        Ok(Value { source: Source::Pin(pin(&node, field)), ty: p.ty })
    }

    fn name(&mut self, name: &str, span: Span) -> Result<Value> {
        if let Some(value) = self.aliases.get(name) {
            return Ok(value.clone());
        }
        // An output is read from the local that holds it.
        if let Some((local, ty)) = self.output(name) {
            let node = self.node("get", NodeKind::Get { var: local.clone(), scope: Scope::Local, class: None });
            return Ok(Value { source: Source::Pin(pin(&node, &local)), ty });
        }
        if let Some(ty) = self.local(name) {
            let node = self.node("get", NodeKind::Get { var: name.to_string(), scope: Scope::Local, class: None });
            return Ok(Value { source: Source::Pin(pin(&node, name)), ty });
        }
        if let Some(ty) = self.params.get(name) {
            let ty = ty.clone();
            return Ok(Value { source: Source::Pin(pin(&self.id("entry"), name)), ty });
        }
        if name == "self" {
            let Some(class) = self.program.class.clone() else {
                bail!("{span}: a function library has no self");
            };
            let node = self.node("self", NodeKind::SelfRef);
            return Ok(Value { source: Source::Pin(pin(&node, "self")), ty: Ty::Object(class) });
        }
        if self.program.dispatchers.contains_key(name) {
            bail!("{span}: {name} is a dispatcher, not a value; call {name}.broadcast()");
        }
        if let Some(ty) = self.program.members.get(name) {
            let ty = ty.clone();
            let node = self.node("get", NodeKind::Get { var: name.to_string(), scope: Scope::SelfMember, class: None });
            return Ok(Value { source: Source::Pin(pin(&node, name)), ty });
        }
        if let Some(p) = self.db().property(&self.program.parent, name) {
            let class = self.db().property_owner(&self.program.parent, name);
            let node = self.node("get", NodeKind::Get { var: name.to_string(), scope: Scope::Inherited, class });
            return Ok(Value { source: Source::Pin(pin(&node, name)), ty: p.ty });
        }
        match self.program.class(name).map_err(|e| anyhow!("{span}: {e}"))? {
            // A class name on its own is the class, e.g. an argument to GetActorOfClass.
            Some(path) => Ok(Value { source: Source::Literal(path.clone()), ty: Ty::Class(path) }),
            None if self.program.functions.contains_key(name) || self.db().has_function(&self.program.parent, name) => {
                bail!("{span}: {name} is a function; call it, {name}(), or pass it where a delegate is expected")
            }
            None => bail!("{span}: no variable {name}"),
        }
    }
}

/// A chain of reads such as `a.b[i].c` as the path `a`, `b`, `[]`, `c`, when it makes no call;
/// none for anything else.
fn read_path(e: &Expr) -> Option<Vec<String>> {
    match &e.kind {
        ExprKind::Name(n) => Some(vec![n.clone()]),
        ExprKind::Member(x, field) => read_path(x).map(|mut p| {
            p.push(field.clone());
            p
        }),
        ExprKind::Index(x, _) => read_path(x).map(|mut p| {
            p.push("[]".to_string());
            p
        }),
        _ => None,
    }
}

/// Whether a method of that name changes the array, set or map it is called on: it is a function
/// of one of their libraries that is not pure. Called on anything else, it only makes a copy of
/// what it is called on where a view would do.
fn changes_container(db: &TypeDb, name: &str) -> bool {
    [Ty::Array(Box::new(Ty::Int)), Ty::Set(Box::new(Ty::Int)), Ty::Map(Box::new(Ty::Int), Box::new(Ty::Int))]
        .iter()
        .any(|on| db.container_function(on, name).ok().flatten().is_some_and(|(f, _)| !f.pure))
}

/// Adds every path the statements assign to or change in place: `x = ..`, `x.f = ..`,
/// `x[i] = ..`, and the methods that change an array or a map, such as `x.Add(..)`.
fn changed_in(body: &[Stmt], out: &mut Vec<Vec<String>>, changes: &dyn Fn(&str) -> bool) {
    fn expr(e: &Expr, out: &mut Vec<Vec<String>>, changes: &dyn Fn(&str) -> bool) {
        match &e.kind {
            ExprKind::Method(object, name, args) => {
                if changes(name) {
                    // an array's, a set's or a map's own methods, which change it in place
                    out.extend(read_path(object));
                }
                expr(object, out, changes);
                args.iter().for_each(|a| expr(a, out, changes));
            }
            ExprKind::Call(_, args, _) | ExprKind::ArrayLiteral(args) => args.iter().for_each(|a| expr(a, out, changes)),
            ExprKind::Make(_, fields) => fields.iter().for_each(|(_, v)| expr(v, out, changes)),
            ExprKind::MapLiteral(entries) => entries.iter().for_each(|(k, v)| {
                expr(k, out, changes);
                expr(v, out, changes)
            }),
            ExprKind::Member(x, _) | ExprKind::Unary(_, x) => expr(x, out, changes),
            ExprKind::Index(a, b) | ExprKind::Binary(_, a, b) => {
                expr(a, out, changes);
                expr(b, out, changes)
            }
            ExprKind::If(cond, a, b) => [cond, a, b].into_iter().for_each(|x| expr(x, out, changes)),
            ExprKind::Match(value, arms, otherwise) => {
                expr(value, out, changes);
                arms.iter().for_each(|(_, arm)| expr(arm, out, changes));
                expr(otherwise, out, changes);
            }
            _ => {}
        }
    }
    for stmt in body {
        match stmt {
            Stmt::Let { value, .. } => expr(value, out, changes),
            Stmt::Assign { target, value, .. } => {
                out.extend(read_path(target));
                expr(target, out, changes);
                expr(value, out, changes);
            }
            Stmt::Return { value, .. } => value.iter().for_each(|v| expr(v, out, changes)),
            Stmt::Break { .. } | Stmt::Continue { .. } => {}
            Stmt::Labelled { looped, .. } => changed_in(std::slice::from_ref(looped), out, changes),
            Stmt::While { cond, body, .. } => {
                expr(cond, out, changes);
                changed_in(body, out, changes);
            }
            Stmt::For { items, body, .. } => {
                expr(items, out, changes);
                changed_in(body, out, changes);
            }
            Stmt::Count { first, last, body, .. } => {
                expr(first, out, changes);
                expr(last, out, changes);
                changed_in(body, out, changes);
            }
            Stmt::Compound { target, value, .. } => {
                out.extend(read_path(target));
                expr(target, out, changes);
                expr(value, out, changes);
            }
            Stmt::If { cond, then, otherwise, .. } => {
                expr(cond, out, changes);
                changed_in(then, out, changes);
                changed_in(otherwise, out, changes);
            }
            Stmt::Match { value, arms, otherwise, .. } => {
                expr(value, out, changes);
                arms.iter().for_each(|(_, body)| changed_in(body, out, changes));
                otherwise.iter().for_each(|body| changed_in(body, out, changes));
            }
            Stmt::Expr(e) => expr(e, out, changes),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::syntax::{parse, parse_source};

    pub fn db() -> TypeDb {
        TypeDb::parse(crate::fixtures::SMALL).unwrap()
    }

    fn compile(src: &str) -> Result<Document> {
        lower(&parse(src)?, "test.lm", &db())
    }

    const HEADER: &str = "blueprint B : Actor at /Game/B\n";

    fn links_into(doc: &Document, node: &str, pin_name: &str) -> Vec<String> {
        doc.ops
            .iter()
            .filter_map(|op| match op {
                Op::Link { from, to } if to.node == node && to.pin == pin_name => Some(from.to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn rejects_mixed_arithmetic() {
        let err = compile(&format!("{HEADER}fn F() {{\n let n = 1\n let x = n + 2.5\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("mixes types"), "{err}");
    }

    #[test]
    fn takes_a_whole_number_beside_a_float_as_a_float() {
        compile(&format!("{HEADER}fn F(x: float) -> bool {{\n return x > 0 && 1 < x * 2\n}}\n")).unwrap();
    }

    #[test]
    fn rejects_a_non_bool_condition() {
        let err = compile(&format!("{HEADER}fn F() {{\n while 1 {{\n }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("not bool"), "{err}");
    }

    #[test]
    fn an_if_joins_both_branches() {
        let doc = compile(&format!("{HEADER}var X: int = 0\nfn F(b: bool) {{\n if b {{\n X = 1\n }}\n X = 2\n}}\n")).unwrap();
        assert_eq!(links_into(&doc, "F/set3", "execute").len(), 2);
    }

    #[test]
    fn an_impure_call_in_a_loop_condition_runs_on_every_pass() {
        let src = format!("{HEADER}fn Next() -> bool {{\n return false\n}}\nfn F() {{\n while Next() {{\n }}\n}}\n");
        let doc = compile(&src).unwrap();
        // The call runs before the branch, and the loop's back edge goes to the call, not the branch.
        assert_eq!(links_into(&doc, "F/call1", "execute"), vec!["F/entry.then", "F/while2.then"]);
    }

    #[test]
    fn a_static_call_takes_its_output_type_from_the_class_argument() {
        let src = format!("{HEADER}fn F() {{\n let a = GameplayStatics.GetActorOfClass(Actor)\n a.K2_DestroyActor()\n}}\n");
        compile(&src).unwrap();
    }

    #[test]
    fn optional_arguments_may_be_left_out() {
        compile(&format!("{HEADER}fn F() {{\n KismetSystemLibrary.PrintString(\"hi\")\n}}\n")).unwrap();
    }

    #[test]
    fn a_pure_call_cannot_stand_alone() {
        let err = compile(&format!("{HEADER}fn F() {{\n KismetMathLibrary.Add_IntInt(1, 2)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("pure"), "{err}");
    }

    #[test]
    fn code_after_a_return_is_rejected() {
        let err = compile(&format!("{HEADER}fn F() -> int {{\n return 1\n return 2\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("never run"), "{err}");
    }

    #[test]
    fn a_break_joins_the_end_of_the_loop() {
        let src = format!("{HEADER}var X: int = 0\nfn F(b: bool) {{\n while b {{\n if b {{\n break\n }}\n X = 1\n }}\n X = 2\n}}\n");
        let doc = compile(&src).unwrap();
        assert_eq!(links_into(&doc, "F/set4", "execute"), vec!["F/while1.else", "F/if2.then"]);
    }

    #[test]
    fn a_continue_goes_on_to_the_next_item() {
        // The body's own end is never reached, so only the continue reaches the increment.
        let src = format!("{HEADER}var Xs: array<int>\nvar X: int = 0\nfn F() {{\n for x in Xs {{\n X = x\n continue\n }}\n}}\n");
        let doc = compile(&src).unwrap();
        assert!(doc.to_json().contains("Add_IntInt"));
    }

    #[test]
    fn break_and_continue_are_only_allowed_inside_a_loop() {
        let err = compile(&format!("{HEADER}fn F() {{\n break\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("only allowed inside a loop"), "{err}");
        let err = compile(&format!("{HEADER}fn F(b: bool) {{\n while b {{\n continue\n return\n }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("never run"), "{err}");
    }

    const LIBRARY: &str = "blueprint L : BlueprintFunctionLibrary at /Game/L\n\
        fn Push(xs: ref array<int>, n: int) {\n xs.Add(n)\n}\n";

    /// Compiles a library and a Blueprint that uses it together, returning the Blueprint's ops.
    fn compile_with_library(src: &str) -> Result<Document> {
        let files = [parse_source(LIBRARY)?, parse_source(src)?];
        let mut db = db();
        declare(&files, &mut db)?;
        lower_source(&files[0], "l.lm", &db)?;
        lower_source(&files[1], "test.lm", &db)
    }

    /// Compiles sources together, returning each one's ops as JSON.
    pub(crate) fn compile_together(sources: &[&str]) -> Result<Vec<String>> {
        let files: Vec<syntax::Source> = sources.iter().map(|src| parse_source(src)).collect::<Result<_>>()?;
        let mut db = db();
        declare(&files, &mut db)?;
        files.iter().map(|file| Ok(lower_source(file, "test.lm", &db)?.to_json())).collect()
    }

    #[test]
    fn a_class_written_by_path_is_checked_when_known() {
        let wrong = compile(&format!("{HEADER}var K: class<Actor> = /Game/Lib/Copies.Copies_C\n")).unwrap_err().to_string();
        assert!(wrong.contains("Copies_C is not a Actor"), "{wrong}");
        compile(&format!("{HEADER}var K: class<ActorComponent> = /Game/Lib/Copies.Copies_C\n")).unwrap();
        compile(&format!("{HEADER}var K: class<Actor> = /Game/Unknown/BP_X.BP_X_C\n")).unwrap();
    }

    #[test]
    fn a_custom_event_is_in_the_event_graph_may_wait_and_is_called_as_a_function() {
        let src = format!("{HEADER}var Lit: bool\nevent Pulse(times: int) {{\n    let n = 0\n    while n < times {{\n        Lit = !Lit\n        KismetSystemLibrary.Delay(0.5)\n        n += 1\n    }}\n}}\nfn Start() {{\n    Pulse(3)\n}}\n");
        let doc = compile(&src).unwrap();
        let event = doc.ops.iter().find_map(|op| match op {
            Op::Node { graph, kind: NodeKind::CustomEvent { name, params }, .. } if name == "Pulse" => Some((graph.clone(), params.len())),
            _ => None,
        });
        assert_eq!(event, Some((EVENT_GRAPH.to_string(), 1)), "the event's node, in the event graph, with its parameter");
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Node { graph, kind: NodeKind::Call { function }, .. } if graph == "Start" && function == "self:Pulse")));
        // Its let names are variables of the Blueprint, private to it, as a waiting handler's are.
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Variable { name, private: true, .. } if name == "Loom_Event_Pulse_n")));

        let other = "blueprint BP_Caller : Actor at /Game/X/BP_Caller\nvar Target: BP_Lamp\nfn Go() {\n    Target.Pulse(2)\n}\n";
        let lamp = "blueprint BP_Lamp : Actor at /Game/X/BP_Lamp\nevent Pulse(times: int) {\n}\n";
        compile_together(&[other, lamp]).unwrap();

        for (decl, want) in [
            ("event Pulse(x: ref int) {\n}", "cannot be ref"),
            ("event Pulse(out x: int) {\n}", "cannot be out"),
            ("event Pulse() -> int {\n}", "an event gives nothing back"),
            ("event cast() {\n}", "cast is a built-in function"),
            ("event Pulse() {\n}\nfn Pulse() {\n}", "Pulse is declared twice"),
        ] {
            let err = compile(&format!("{HEADER}{decl}\n")).unwrap_err().to_string();
            assert!(err.contains(want), "{decl}: {err}");
        }
    }

    #[test]
    fn a_labelled_break_and_continue_leave_or_go_on_with_the_loop_they_name() {
        let src = format!("{HEADER}fn F(n: int) -> int {{\n    let hits = 0\n    outer: for i in 1..n {{\n        inner: for j in 1..n {{\n            if j > i {{\n                continue outer\n            }}\n            if i * j > 50 {{\n                break outer\n            }}\n            if j == 3 {{\n                break inner\n            }}\n            hits += 1\n        }}\n    }}\n    return hits\n}}\n");
        compile(&src).unwrap();
        for (body, expected) in [
            ("while true {\n        break nowhere\n    }", "no loop around this break is labelled nowhere"),
            ("again: while true {\n        again: while true {\n            break\n        }\n    }", "labelled again already"),
        ] {
            let err = compile(&format!("{HEADER}fn F() {{\n    {body}\n}}\n")).unwrap_err().to_string();
            assert!(err.contains(expected), "{err}");
        }
        let err = parse(&format!("{HEADER}fn F() {{\n    here: return\n}}\n")).unwrap_err().to_string();
        assert!(err.contains("a label names a loop"), "{err}");
    }

    #[test]
    fn a_private_variable_is_read_and_set_only_by_its_own_blueprint() {
        let owner = "blueprint BP_Safe : Actor at /Game/X/BP_Safe\nprivate var Code: int = 4\nfn Open() -> int {\n    Code = Code + 1\n    return Code\n}\n";
        let docs = compile_together(&[owner]).unwrap();
        assert!(docs[0].contains("\"private\":true"), "{}", docs[0]);
        let thief = "blueprint BP_Thief : Actor at /Game/X/BP_Thief\nvar Safe: BP_Safe\nfn Peek() -> int {\n    return Safe.Code\n}\n";
        let err = compile_together(&[thief, owner]).unwrap_err().to_string();
        assert!(err.contains("Code is private to BP_Safe"), "{err}");
        let heir = "blueprint BP_Heir : BP_Safe at /Game/X/BP_Heir\nfn Peek() -> int {\n    return Code\n}\n";
        compile_together(&[heir, owner]).unwrap_err();
        let err = compile(&format!("{HEADER}private editable var X: int\n")).unwrap_err().to_string();
        assert!(err.contains("expected 'fn'"), "{err}");
    }

    #[test]
    fn a_function_with_a_built_ins_name_is_declared_and_called_in_backticks() {
        let err = compile(&format!("{HEADER}fn cast() {{\n}}\n")).unwrap_err().to_string();
        assert!(err.contains("cast is a built-in function; write `cast` in backticks"), "{err}");
        // In backticks it is the Blueprint's own; bare, it is still the built-in.
        let doc = compile(&format!("{HEADER}var A: Actor\nfn `cast`() -> int {{\n    return 1\n}}\nfn F() -> int {{\n    if cast(A, ActorComponent) != none {{\n        return `cast`()\n    }}\n    return 0\n}}\n")).unwrap();
        let json = doc.to_json();
        assert!(json.contains("\"cast\""), "the own function is called: {json}");
        assert!(json.contains("ActorComponent"), "the built-in cast is still made: {json}");
    }

    #[test]
    fn another_blueprints_outputs_are_read_from_its_record() {
        let counter = "blueprint BP_Counter : Actor at /Game/X/BP_Counter\nfn Count(out Total: int, out Seen: bool) {\n Total = 3\n Seen = true\n}\n";
        let user = "blueprint BP_User : Actor at /Game/X/BP_User\nvar Counter: BP_Counter\nfn F() -> int {\n let r = Counter.Count()\n if r.Seen {\n  return r.Total\n }\n return 0\n}\n";
        compile_together(&[user, counter]).unwrap();
        let wrong = "blueprint BP_User : Actor at /Game/X/BP_User\nvar Counter: BP_Counter\nfn F() -> string {\n let r = Counter.Count()\n return r.Total\n}\n";
        compile_together(&[wrong, counter]).unwrap_err();
    }

    #[test]
    fn array_methods_find_and_remove_items() {
        let src = format!("{HEADER}var Items: array<int>\nfn F() -> bool {{\n Items.Remove(0)\n if Items.Contains(2) && Items.Find(3) >= 0 {{\n  return Items.RemoveItem(2)\n }}\n return false\n}}\n");
        let json = compile(&src).unwrap().to_json();
        for function in ["Array_Remove", "Array_Contains", "Array_Find", "Array_RemoveItem"] {
            assert!(json.contains(&format!(":{function}\"")), "{function} missing from {json}");
        }
        for pin in ["IndexToRemove", "ItemToFind", ".Item\""] {
            assert!(json.contains(pin), "{pin} missing");
        }
        let err = compile(&format!("{HEADER}var Items: array<int>\nfn F() {{\n Items.RemoveItem(\"a\")\n}}\n")).unwrap_err().to_string();
        assert!(err.contains("Item is int, but the value is string"), "{err}");
    }

    #[test]
    fn a_blueprint_may_come_before_the_source_of_its_parent() {
        let child = "blueprint BP_Child : BP_Parent at /Game/X/BP_Child\nfn F() -> int {\n return Seen\n}\n";
        let parent = "blueprint BP_Parent : Actor at /Game/X/BP_Parent\nvar Seen: int = 1\n";
        compile_together(&[child, parent]).unwrap();

        let a = "blueprint BP_A : BP_B at /Game/X/BP_A\n";
        let b = "blueprint BP_B : BP_A at /Game/X/BP_B\n";
        let err = compile_together(&[a, b]).unwrap_err().to_string();
        assert!(err.contains("BP_A: its parent BP_B comes back round to it"), "{err}");
        let lost = "blueprint BP_Lost : BP_Nowhere at /Game/X/BP_Lost\n";
        let err = compile_together(&[lost]).unwrap_err().to_string();
        assert!(err.contains("the parent class BP_Nowhere is not in the types"), "{err}");
    }

    #[test]
    fn a_short_name_finds_the_blueprint_in_the_folders_a_source_uses() {
        let one = "blueprint BP_Inv : Actor at /Game/Lib_1/BP_Inv\n";
        let two = "blueprint BP_Inv : Actor at /Game/Lib_2/BP_Inv\n";
        let deep = "blueprint BP_Row : Actor at /Game/Lib_2/UI/BP_Row\n";
        let shop = |uses: &str| format!("blueprint BP_Shop : Actor at /Game/Shop/BP_Shop\n{uses}var Stock: BP_Inv\n");
        let used = |uses: &str| compile_together(&[one, two, deep, &shop(uses)]).map(|docs| docs[3].clone());

        assert!(used("use /Game/Lib_1\n").unwrap().contains("/Game/Lib_1/BP_Inv.BP_Inv_C"));
        assert!(used("use /Game/Lib_2\n").unwrap().contains("/Game/Lib_2/BP_Inv.BP_Inv_C"));
        // The first folder written that has one wins.
        assert!(used("use /Game/Lib_2\nuse /Game/Lib_1\n").unwrap().contains("/Game/Lib_2/BP_Inv.BP_Inv_C"));
        let err = used("").unwrap_err().to_string();
        assert!(err.contains("BP_Inv is ambiguous"), "{err}");
        // A used folder holds what is in the folders inside it.
        let row = "blueprint BP_Shop : Actor at /Game/Shop/BP_Shop\nuse /Game/Lib_2\nvar Row: BP_Row\n";
        compile_together(&[one, two, deep, row]).unwrap();
        // Two inside one used folder are still ambiguous.
        let other = "blueprint BP_Inv : Actor at /Game/Lib_1/Old/BP_Inv\n";
        let err = compile_together(&[one, other, &shop("use /Game/Lib_1\n")]).unwrap_err().to_string();
        assert!(err.contains("BP_Inv is ambiguous"), "{err}");
    }

    #[test]
    fn a_used_folder_must_hold_something_and_come_first() {
        let err = compile(&format!("{HEADER}use /Game/Nothing\n")).unwrap_err().to_string();
        assert!(err.contains("nothing a source can use is under /Game/Nothing"), "{err}");
        let db = db();
        let file = parse(&format!("{HEADER}use /Game/Nothing\n")).unwrap();
        assert!(lower(&file, "test.lm", &db).is_err());
        assert!(db.missing().contains(&"/Game/Nothing/".to_string()), "{:?}", db.missing());

        let err = parse(&format!("{HEADER}var N: int\nuse /Game/Lib\n")).unwrap_err().to_string();
        assert!(err.contains("use comes before the Blueprint's other items"), "{err}");
        let err = parse(&format!("{HEADER}use /Game/Lib/BP_X.BP_X_C\n")).unwrap_err().to_string();
        assert!(err.contains("use takes a folder"), "{err}");
    }

    #[test]
    fn a_short_name_finds_the_blueprint_in_its_own_folder_first() {
        let one = "blueprint BP_A : Actor at /Game/One/BP_A\n";
        let two = "blueprint BP_A : Actor at /Game/Two/BP_A\n";
        let user_one = "blueprint BP_User : Actor at /Game/One/BP_User\nvar Thing: BP_A\nvar Kind: class<Actor> = BP_A\n";
        let user_two = "blueprint BP_User : Actor at /Game/Two/BP_User\nvar Thing: BP_A\nvar Kind: class<Actor> = BP_A\n";
        let docs = compile_together(&[one, two, user_one, user_two]).unwrap();
        assert!(docs[2].contains("/Game/One/BP_A.BP_A_C") && !docs[2].contains("/Game/Two/BP_A"), "{}", docs[2]);
        assert!(docs[3].contains("/Game/Two/BP_A.BP_A_C") && !docs[3].contains("/Game/One/BP_A"), "{}", docs[3]);

        // From anywhere else, two of one name is a choice the source must make.
        let elsewhere = "blueprint BP_Other : Actor at /Game/Three/BP_Other\nvar Thing: BP_A\n";
        let err = compile_together(&[one, two, elsewhere]).unwrap_err().to_string();
        assert!(err.contains("BP_A is ambiguous; write its path, one of /Game/One/BP_A.BP_A_C, /Game/Two/BP_A.BP_A_C"), "{err}");
    }

    #[test]
    fn a_library_function_is_called_on_its_class() {
        let src = format!("{HEADER}var Xs: array<int>\nfn F() {{\n L.Push(Xs, 1)\n}}\n");
        let doc = compile_with_library(&src).unwrap();
        let call = doc.ops.iter().find_map(|op| match op {
            Op::Node { id, kind: NodeKind::Call { function }, .. } if function.ends_with(":Push") => Some((id.clone(), function.clone())),
            _ => None,
        });
        let (id, function) = call.expect("a call to Push");
        assert_eq!(function, "/Game/L.L_C:Push");
        assert!(links_into(&doc, &id, "self").is_empty(), "a static call has no target");
    }

    #[test]
    fn a_by_reference_parameter_is_marked_and_needs_a_variable() {
        let doc = compile(LIBRARY).unwrap();
        let by_ref = doc.ops.iter().any(|op| matches!(op, Op::Function { params, .. } if params.iter().any(|p| p.name == "xs" && p.by_ref)));
        assert!(by_ref, "xs is passed by reference");
        let err = compile_with_library(&format!("{HEADER}fn F() {{\n L.Push([1], 2)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("must be a variable"), "{err}");
    }

    #[test]
    fn a_value_written_out_for_a_pin_that_takes_none_goes_through_a_local() {
        let doc = compile(&format!("{HEADER}fn F() -> string {{\n return string(Actor)\n}}\n")).unwrap();
        let local = doc.ops.iter().find_map(|op| match op {
            Op::Function { locals, .. } => locals.iter().find(|l| l.name.starts_with("Loom_Value")),
            _ => None,
        });
        let local = local.expect("the class goes into a local");
        assert_eq!(local.ty.to_string(), "class:/Script/Engine.Actor", "the local has the value's type, not the parameter's");
        let written = doc.ops.iter().any(|op| matches!(op, Op::Default { pin, .. } if pin.pin == "Class"));
        assert!(!written, "nothing is written on the conversion's pin");
    }

    #[test]
    fn a_value_written_out_for_a_wired_input_of_any_call_goes_through_a_local() {
        let doc = compile(&format!("{HEADER}fn F() -> bool {{\n return KismetSystemLibrary.IsValidSoftClassReference(/Script/Engine.Actor)\n}}\n")).unwrap();
        let local = doc.ops.iter().any(|op| matches!(op, Op::Function { locals, .. } if locals.iter().any(|l| l.name.starts_with("Loom_Value"))));
        assert!(local, "the reference goes into a local");
        let written = doc.ops.iter().any(|op| matches!(op, Op::Default { pin, .. } if pin.pin == "SoftClassReference"));
        assert!(!written, "nothing is written on the wired pin");
    }

    #[test]
    fn a_structs_field_may_be_a_container() {
        let doc = compile(&format!("{HEADER}var M: LODMappingData = LODMappingData(Mapping: [1, 2])\n")).unwrap();
        let written = doc.ops.iter().any(|op| matches!(op, Op::Variable { default: Some(d), .. } if d == "(Mapping=(1,2))"));
        assert!(written, "{:?}", doc.ops);
    }

    #[test]
    fn a_functions_doc_comment_and_the_category_describe_it() {
        let src = format!("{HEADER}category \"Tools|Things\"\n\n// Unrelated.\n\n// Does a thing,\n// twice.\n//\n// Then stops.\nfn F() {{\n}}\n");
        let doc = compile(&src).unwrap();
        let function = doc.ops.iter().find_map(|op| match op {
            Op::Function { description, category, .. } => Some((description.clone(), category.clone())),
            _ => None,
        });
        assert_eq!(
            function,
            Some((Some("Does a thing, twice.\n\nThen stops.".to_string()), Some("Tools|Things".to_string())))
        );
    }

    #[test]
    fn an_editable_variable_is_marked_and_described() {
        let src = format!("{HEADER}category \"Tools\"\n// Who we are.\neditable var Id: string\nvar Hidden: int\n");
        let doc = compile(&src).unwrap();
        let var = |wanted: &str| {
            doc.ops.iter().find_map(|op| match op {
                Op::Variable { name, editable, description, category, .. } if name == wanted => {
                    Some((*editable, description.clone(), category.clone()))
                }
                _ => None,
            })
        };
        assert_eq!(var("Id"), Some((true, Some("Who we are.".to_string()), Some("Tools".to_string()))));
        assert_eq!(var("Hidden"), Some((false, None, Some("Tools".to_string()))));
        let err = compile(&format!("{HEADER}editable fn F() {{\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("expected 'var'"), "{err}");
    }

    #[test]
    fn a_dispatcher_is_declared_and_broadcast() {
        let src = format!("{HEADER}dispatcher Done(score: int)\nfn F() {{\n Done.broadcast(3)\n}}\n");
        let doc = compile(&src).unwrap();
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Dispatcher { name, params, .. } if name == "Done" && params.len() == 1)));
        let node = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Node { id, kind: NodeKind::Broadcast { dispatcher }, .. } if dispatcher == "Done" => Some(id.clone()),
                _ => None,
            })
            .unwrap();
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Default { pin, value } if pin.node == node && pin.pin == "score" && value == "3")));
        let wrong = |body: &str| compile(&format!("{HEADER}dispatcher Done(score: int)\nfn F() {{\n {body}\n}}\n")).unwrap_err().to_string();
        assert!(wrong("Done.broadcast()").contains("passes (score: int)"));
        assert!(wrong("Done.call(1)").contains("has broadcast(), bind()"));
        assert!(wrong("let x = Done").contains("not a value"));
        let err = compile("blueprint L : BlueprintFunctionLibrary at /Game/L\ndispatcher Done()\n").unwrap_err();
        assert!(err.to_string().contains("no dispatchers"), "{err}");
    }

    const CHECKER: &str = "editor blueprint C : Checker at /Game/C\n";

    #[test]
    fn only_an_editor_blueprint_uses_the_editors_classes() {
        let err = compile("blueprint C : Checker at /Game/C\n").unwrap_err();
        assert!(err.to_string().contains("write `editor blueprint`"), "{err}");
        let err = compile("editor blueprint C : Actor at /Game/C\n").unwrap_err();
        assert!(err.to_string().contains("remove `editor`"), "{err}");
        let err = compile(&format!("{HEADER}fn F() {{\n let s = subsystem(ThingSubsystem)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("only an `editor blueprint`"), "{err}");
        compile(&format!("{CHECKER}fn F() -> int {{\n return subsystem(ThingSubsystem).Count()\n}}\n")).unwrap();
    }

    #[test]
    fn an_event_that_returns_is_overridden_as_a_function() {
        let doc = compile(&format!("{CHECKER}on K2_Check(asset: Object) -> bool {{\n Fail(asset)\n return true\n}}\n")).unwrap();
        let overrides = doc.ops.iter().find_map(|op| match op {
            Op::Function { name, overrides, .. } if name == "K2_Check" => Some(overrides.clone()),
            _ => None,
        });
        assert_eq!(overrides, Some(Some("/Script/Test.Checker".to_string())));
        // It is a function of its own, not an event node calling one.
        assert!(!doc.ops.iter().any(|op| matches!(op, Op::Node { kind: NodeKind::Event { .. }, .. })));
        let err = compile(&format!("{CHECKER}on K2_Check(asset: Object) {{\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("write `-> bool`"), "{err}");
        let err = compile(&format!("{CHECKER}on K2_Check(asset: Object) -> int {{\n return 1\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("returns bool, not int"), "{err}");
        let err = compile(&format!("{HEADER}on ReceiveTick(dt: float) -> bool {{\n return true\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("returns nothing"), "{err}");
    }

    const IMPLEMENTS: &str = "blueprint B : Actor at /Game/B\nimplements DataflowSimulationActor\nimplements MovieSceneEasingFunction\n";

    #[test]
    fn an_interface_is_implemented_before_any_function_and_its_handlers_implement_it() {
        let src = format!("{IMPLEMENTS}var N: int\non PreDataflowSimulationTick(time: float, dt: float) {{\n N = 1\n}}\non OnEvaluate(t: float) -> float {{\n return t\n}}\n");
        let doc = compile(&src).unwrap();
        let implemented: Vec<&str> = doc.ops.iter().filter_map(|op| match op {
            Op::Implement { interface } => Some(interface.as_str()),
            _ => None,
        }).collect();
        assert_eq!(implemented, ["/Script/DataflowSimulation.DataflowSimulationActor", "/Script/MovieScene.MovieSceneEasingFunction"]);
        assert!(matches!(doc.ops[1], Op::Implement { .. }) && matches!(doc.ops[2], Op::Implement { .. }));
        // An event that returns nothing is an event node, named by its interface.
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Node { kind: NodeKind::Event { event, class }, .. }
            if event == "PreDataflowSimulationTick" && class == "/Script/DataflowSimulation.DataflowSimulationActor")));
        // One that returns a value overrides it as a function.
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Function { name, overrides: Some(o), .. }
            if name == "OnEvaluate" && o == "/Script/MovieScene.MovieSceneEasingFunction")));
        // Leaving one out is allowed.
        compile(IMPLEMENTS).unwrap();
    }

    #[test]
    fn an_implements_that_cannot_be_is_refused() {
        let wrong = |src: &str| compile(src).unwrap_err().to_string();
        assert!(wrong(&format!("{HEADER}implements Actor\n")).contains("Actor is not an interface"));
        assert!(wrong(&format!("{HEADER}implements Sealed\n")).contains("a Blueprint cannot implement"));
        assert!(wrong(&format!("{HEADER}implements MovieSceneEasingFunction\nimplements MovieSceneEasingFunction\n")).contains("implemented twice"));
        assert!(wrong("blueprint L : BlueprintFunctionLibrary at /Game/L\nimplements MovieSceneEasingFunction\n").contains("implements no interfaces"));
        let err = parse(&format!("{HEADER}var N: int\nimplements MovieSceneEasingFunction\n")).unwrap_err().to_string();
        assert!(err.contains("implements comes after the header and any use"), "{err}");
        assert!(wrong(&format!("{IMPLEMENTS}on Nothing() {{\n}}\n")).contains("neither has any interface it implements"));
        assert!(wrong(&format!("{IMPLEMENTS}fn OnEvaluate() {{\n}}\n")).contains("implement it with `on OnEvaluate`"));

        let parent = "blueprint BP_P : Actor at /Game/X/BP_P\nimplements MovieSceneEasingFunction\n";
        let child = "blueprint BP_C : BP_P at /Game/X/BP_C\nimplements MovieSceneEasingFunction\n";
        let err = compile_together(&[parent, child]).unwrap_err().to_string();
        assert!(err.contains("BP_P_C already implements MovieSceneEasingFunction"), "{err}");
        // Its handlers implement the parent's interface without it.
        let child = "blueprint BP_C : BP_P at /Game/X/BP_C\non OnEvaluate(t: float) -> float {\n return t\n}\n";
        let docs = compile_together(&[parent, child]).unwrap();
        assert!(docs[1].contains(r#""overrides":"/Script/MovieScene.MovieSceneEasingFunction""#), "{}", docs[1]);
    }

    #[test]
    fn an_interfaces_function_is_called_by_message() {
        let src = format!("{HEADER}fn F(thing: Object) -> float {{\n return cast(thing, MovieSceneEasingFunction).OnEvaluate(0.5)\n}}\n");
        let doc = compile(&src).unwrap();
        let message = doc.ops.iter().find_map(|op| match op {
            Op::Node { id, kind: NodeKind::Message { function }, .. } => Some((id.clone(), function.clone())),
            _ => None,
        });
        let (id, function) = message.expect("a message node");
        assert_eq!(function, "/Script/MovieScene.MovieSceneEasingFunction:OnEvaluate");
        // It runs in order, though the function is pure, and takes the object the interface is.
        assert_eq!(links_into(&doc, &id, "execute"), ["F/entry.then"]);
        assert!(links_into(&doc, &id, "self")[0].contains("pure"), "the object comes from Conv_InterfaceToObject");

        // Without a cast, on an object whose class implements the interface.
        compile(&format!("{IMPLEMENTS}fn F() -> float {{\n return OnEvaluate(1) + self.OnEvaluate(2)\n}}\n")).unwrap();
        let err = compile(&format!("{HEADER}fn F(thing: Object) -> float {{\n return thing.OnEvaluate(1)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("Object has no function OnEvaluate"), "{err}");

        let wrong = |body: &str| compile(&format!("{HEADER}fn F(thing: Object) -> bool {{\n {body}\n}}\n")).unwrap_err().to_string();
        assert!(wrong("return implements(thing, Actor)").contains("Actor is not an interface"));
        assert!(wrong("return implements(1, MovieSceneEasingFunction)").contains("takes an object"));
    }

    #[test]
    fn a_cast_is_pure_and_typed_by_its_class() {
        let doc = compile(&format!("{HEADER}fn F(o: Object) -> Actor {{\n return cast(o, Actor)\n}}\n")).unwrap();
        let node = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Node { id, kind: NodeKind::Cast { class }, .. } if class == "/Script/Engine.Actor" => Some(id.clone()),
                _ => None,
            })
            .unwrap();
        assert!(!links_into(&doc, &node, "Object").is_empty());
        // Pure: nothing runs it.
        assert!(links_into(&doc, &node, "execute").is_empty());
        let err = compile(&format!("{HEADER}fn F() {{\n let a = cast(1, Actor)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("takes an object"), "{err}");
    }

    #[test]
    fn a_component_is_a_member_with_its_fields_set() {
        let src = format!("{HEADER}component C: Copies(Id: \"example\")\nfn F() -> bool {{\n return C.Running\n}}\n");
        let doc = compile(&src).unwrap();
        let fields = doc.ops.iter().find_map(|op| match op {
            Op::Component { name, class, fields } if name == "C" && class == "/Game/Lib/Copies.Copies_C" => Some(fields.clone()),
            _ => None,
        });
        assert_eq!(fields, Some(vec![ComponentField { name: "Id".into(), value: "example".into() }]));
        let wrong = |line: &str| compile(&format!("{HEADER}{line}\n")).unwrap_err().to_string();
        assert!(wrong("component C: Copies(Nope: 1)").contains("has no Nope"));
        assert!(wrong("component C: Pieces").contains("outside the project's InstancedClassPaths (/Game/Lib)"));
        assert!(wrong("component C: Actor").contains("not a component"));
        let err = compile("blueprint L : BlueprintFunctionLibrary at /Game/L\ncomponent C: Copies\n").unwrap_err();
        assert!(err.to_string().contains("only an actor's"), "{err}");
    }

    #[test]
    fn a_components_dispatcher_is_handled_by_a_bound_event() {
        let src = format!("{HEADER}component C: Copies\nvar Last: int\non C.OnRun(copy: int) {{\n Last = copy\n}}\n");
        let doc = compile(&src).unwrap();
        let node = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Node { id, kind: NodeKind::BoundEvent { component, dispatcher }, .. } if component == "C" && dispatcher == "OnRun" => {
                    Some(id.clone())
                }
                _ => None,
            })
            .unwrap();
        // Its parameter reaches the handler, which is a function of its own.
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Link { from, .. } if from.node == node && from.pin == "Copy")));
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Function { name, .. } if name == "Loom_On_C_OnRun")));
        let wrong = |body: &str| compile(&format!("{HEADER}component C: Copies\n{body}\n")).unwrap_err().to_string();
        assert!(wrong("on C.OnRun {\n}").contains("passes (Copy: int)"));
        assert!(wrong("on C.OnWalk {\n}").contains("no dispatcher OnWalk"));
        assert!(wrong("on D.OnRun {\n}").contains("not a component"));
    }

    #[test]
    fn a_library_has_only_functions() {
        let err = compile("blueprint L : BlueprintFunctionLibrary at /Game/L\nvar X: int = 0\n").unwrap_err();
        assert!(err.to_string().contains("no variables"), "{err}");
    }

    #[test]
    fn a_library_derives_from_no_other_library() {
        let a = "blueprint BPL_A : BlueprintFunctionLibrary at /Game/L/BPL_A\npure fn F() -> int {\n return 1\n}\n";
        let b = "blueprint BPL_B : BPL_A at /Game/L/BPL_B\n";
        let err = compile_together(&[a, b]).unwrap_err().to_string();
        assert!(err.contains("BPL_B: a function library can't derive from another library, and BPL_A is one"), "{err}");
        let err = compile("blueprint L : KismetMathLibrary at /Game/L\n").unwrap_err().to_string();
        assert!(err.contains("KismetMathLibrary is one"), "{err}");
        compile("blueprint L : BlueprintFunctionLibrary at /Game/L\n").unwrap();
    }

    #[test]
    fn a_handler_that_waits_is_lowered_into_the_event_graph() {
        let src = format!("{HEADER}var N: int\non ReceiveTick(dt: float) {{\n let twice = dt * 2\n N = 1\n if dt > 1.0 {{\n return\n }}\n KismetSystemLibrary.Delay(twice)\n N = 2\n}}\n");
        let doc = compile(&src).unwrap();
        // No function of its own: the event node starts the chain, and the wait is in the event graph.
        assert!(!doc.ops.iter().any(|op| matches!(op, Op::Function { name, .. } if name == "Loom_On_ReceiveTick")));
        let event = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Node { id, graph, kind: NodeKind::Event { event, .. } } if event == "ReceiveTick" && graph == EVENT_GRAPH => Some(id.clone()),
                _ => None,
            })
            .unwrap();
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Link { from, .. } if from.node == event && from.pin == "DeltaSeconds")));
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Node { graph, kind: NodeKind::Call { function }, .. }
            if graph == EVENT_GRAPH && function.ends_with(":Delay"))));
        // Its local is a variable of the Blueprint, read and set by its own name.
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Variable { name, .. } if name == "Loom_On_ReceiveTick_twice")));
        assert!(!doc.ops.iter().any(|op| matches!(op, Op::Node { kind: NodeKind::Get { scope: Scope::Local, .. } | NodeKind::Set { scope: Scope::Local, .. }, .. })));
        let set = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Node { id, kind: NodeKind::Set { var, .. }, .. } if var == "Loom_On_ReceiveTick_twice" => Some(id.clone()),
                _ => None,
            })
            .unwrap();
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Link { to, .. } if to.node == set && to.pin == "Loom_On_ReceiveTick_twice")));
        // A handler that doesn't wait is still a function.
        let doc = compile(&format!("{HEADER}var N: int\non ReceiveTick(dt: float) {{\n N = 1\n}}\n")).unwrap();
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Function { name, .. } if name == "Loom_On_ReceiveTick")));
    }

    #[test]
    fn a_components_handler_can_wait() {
        let src = format!("{HEADER}component C: Copies\nvar Last: int\non C.OnRun(copy: int) {{\n KismetSystemLibrary.Delay(0.5)\n Last = copy\n}}\n");
        let doc = compile(&src).unwrap();
        assert!(!doc.ops.iter().any(|op| matches!(op, Op::Function { name, .. } if name == "Loom_On_C_OnRun")));
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Link { from, .. } if from.pin == "Copy")));
    }

    #[test]
    fn only_an_events_handler_can_wait() {
        let wait = "KismetSystemLibrary.Delay(1.0)";
        let err = compile(&format!("{HEADER}fn F() {{\n {wait}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("Delay waits, so it can only be called in an event's `on` handler"), "{err}");
        let err = compile(&format!("{CHECKER}on K2_Check(asset: Object) -> bool {{\n {wait}\n return true\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("Delay waits"), "{err}");
    }

    #[test]
    fn an_if_expression_computes_only_the_side_it_chooses() {
        let src = format!("{HEADER}var N: int\nfn Next() -> int {{\n N = N + 1\n return N\n}}\nfn F(ok: bool) -> int {{\n return if ok {{ Next() }} else {{ 0 }}\n}}\n");
        let doc = compile(&src).unwrap();
        // The call is on the branch's then side, not before the branch.
        let call = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Node { id, graph, kind: NodeKind::Call { function } } if graph == "F" && function == "self:Next" => Some(id.clone()),
                _ => None,
            })
            .unwrap();
        let from = links_into(&doc, &call, "execute");
        assert_eq!(from.len(), 1);
        assert!(from[0].starts_with("F/if") && from[0].ends_with(".then"), "{from:?}");
        // Each side sets the if's own local, which is then read.
        let sets = doc.ops.iter().filter(|op| matches!(op, Op::Node { kind: NodeKind::Set { var, scope: Scope::Local, .. }, .. } if var.starts_with("Loom_If"))).count();
        assert_eq!(sets, 2);
    }

    #[test]
    fn an_if_expressions_sides_agree_on_a_type() {
        compile(&format!("{HEADER}fn F(fast: bool) -> float {{\n return if fast {{ 600.0 }} else {{ 300 }}\n}}\n")).unwrap();
        compile(&format!("{HEADER}fn F(fast: bool) -> float {{\n let speed = if fast {{ 600 }} else {{ 300.5 }}\n return speed\n}}\n")).unwrap();
        compile(&format!("{HEADER}fn F(fast: bool) -> float {{\n let speed: float = if fast {{ 600 }} else {{ 300 }}\n return speed\n}}\n")).unwrap();
        compile(&format!("{HEADER}fn F(n: int) -> string {{\n return if n == 0 {{ \"none\" }} else if n == 1 {{ \"one\" }} else {{ \"many\" }}\n}}\n")).unwrap();
        compile(&format!("{HEADER}fn F(ok: bool) -> Actor {{\n let a = if ok {{ self }} else {{ none }}\n return a\n}}\n")).unwrap();
        let err = compile(&format!("{HEADER}fn F(ok: bool) {{\n let x = if ok {{ 1 }} else {{ \"one\" }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("one side is int and the other string"), "{err}");
        let err = compile(&format!("{HEADER}fn F(ok: bool) {{\n let x = if ok {{ none }} else {{ none }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("needs a type"), "{err}");
        let err = compile(&format!("{HEADER}fn F(ok: bool) -> int {{\n return if ok {{ 1 }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("needs an `else`"), "{err}");
        let err = compile(&format!("{HEADER}fn F(ok: bool, n: int) -> bool {{\n return ok && (if n > 0 {{ true }} else {{ false }})\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("or an `if`"), "{err}");
    }

    #[test]
    fn an_if_expression_in_a_handler_that_waits_keeps_its_value_in_a_variable() {
        let src = format!("{HEADER}var N: int\non ReceiveTick(dt: float) {{\n let x = if dt > 1.0 {{ 1 }} else {{ 2 }}\n KismetSystemLibrary.Delay(1.0)\n N = x\n}}\n");
        let doc = compile(&src).unwrap();
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Variable { name, .. } if name.starts_with("Loom_On_ReceiveTick_Loom_If"))));
    }

    #[test]
    fn a_match_runs_the_first_arm_that_lists_the_value() {
        let src = format!("{HEADER}var N: int\nfn F(g: ETickingGroup) {{\n match g {{\n  ETickingGroup.TG_PrePhysics, ETickingGroup.TG_StartPhysics {{\n   N = 1\n  }}\n  ETickingGroup.TG_EndPhysics {{\n   N = 2\n  }}\n  else {{\n   N = 3\n  }}\n }}\n}}\n");
        let doc = compile(&src).unwrap();
        // The value is held once, and each arm is a branch on it.
        let held = doc.ops.iter().filter(|op| matches!(op, Op::Node { kind: NodeKind::Set { var, .. }, .. } if var.starts_with("Loom_Match"))).count();
        assert_eq!(held, 1);
        let branches = doc.ops.iter().filter(|op| matches!(op, Op::Node { kind: NodeKind::Branch, .. })).count();
        assert_eq!(branches, 2);
        compile(&format!("{HEADER}fn F(n: int) {{\n match n {{\n  -1, 0 {{\n   return\n  }}\n  1 {{\n  }}\n }}\n}}\n")).unwrap();
        compile(&format!("{HEADER}fn F(s: string) {{\n match s {{\n  \"a\" {{\n  }}\n }}\n}}\n")).unwrap();
    }

    #[test]
    fn a_match_lists_each_value_written_out_once() {
        let wrong = |body: &str| compile(&format!("{HEADER}var M: int\nfn F(n: int, f: float) {{\n{body}\n}}\n")).unwrap_err().to_string();
        assert!(wrong("match n {\n 1 {\n }\n 2, 1 {\n }\n}").contains("1 is listed by an earlier arm"));
        assert!(wrong("match n {\n M {\n }\n}").contains("written out"));
        assert!(wrong("match f {\n 1 {\n }\n}").contains("not float"));
        assert!(wrong("match n {\n else {\n }\n 1 {\n }\n}").contains("comes last"));
        assert!(wrong("match n {\n \"a\" {\n }\n}").contains("mixes types"));
    }

    #[test]
    fn a_match_expression_gives_the_chosen_arms_value() {
        compile(&format!("{HEADER}fn F(g: ETickingGroup) -> float {{\n return match g {{ ETickingGroup.TG_PrePhysics {{ 1.5 }} ETickingGroup.TG_EndPhysics {{ 2 }} else {{ 0 }} }}\n}}\n")).unwrap();
        compile(&format!("{HEADER}fn F(n: int) -> string {{\n let s = match n {{\n  1 {{ \"one\" }}\n  else {{ \"many\" }}\n }}\n return s\n}}\n")).unwrap();
        let err = compile(&format!("{HEADER}fn F(n: int) -> int {{\n return match n {{ 1 {{ 1 }} }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("needs an `else` arm"), "{err}");
    }

    const TOOL: &str = "// What a worker holds.\nenum E_Tool at /Game/Types/E_Tool {\n    Hammer\n    // Cuts wood.\n    Saw \"Hand Saw\"\n}\n";

    #[test]
    fn an_enum_is_made_with_its_values_in_order() {
        let [made, _] = &compile_together(&[TOOL, &format!("{HEADER}var T: E_Tool\n")]).unwrap()[..] else { panic!() };
        assert!(made.contains(r#"{"op":"enum","path":"/Game/Types/E_Tool","source":"test.lm","values":[{"name":"Hammer"},{"name":"Saw","label":"Hand Saw","description":"Cuts wood."}],"description":"What a worker holds."}"#), "{made}");
        assert!(made.contains(r#"{"op":"save"}"#), "{made}");
    }

    #[test]
    fn a_blueprint_uses_an_enum_declared_beside_it() {
        let src = format!("{HEADER}use /Game/Types\nvar T: E_Tool = E_Tool.Saw\nfn Cuts() -> bool {{\n return match T {{ E_Tool.Saw {{ true }} else {{ false }} }}\n}}\n");
        compile_together(&[TOOL, &src]).unwrap();
        let err = compile_together(&[TOOL, &format!("{HEADER}use /Game/Types\nvar T: E_Tool = E_Tool.Axe\n")]).unwrap_err();
        assert!(err.to_string().contains("Axe"), "{err}");
    }

    #[test]
    fn an_enum_lists_each_value_once_and_shows_each_apart() {
        for (values, want) in [
            ("", "has no values"),
            ("    A\n    A\n", "A is listed twice"),
            ("    A\n    B \"A\"\n", "B and A would both be shown as \"A\""),
            ("    A \" \"\n", "A's label is empty"),
        ] {
            let err = compile_together(&[&format!("enum E at /Game/E {{\n{values}}}\n")]).unwrap_err();
            assert!(err.to_string().contains(want), "{values}: {err}");
        }
        let err = compile_together(&["enum E at /Game/F {\n    A\n}\n"]).unwrap_err();
        assert!(err.to_string().contains("must be saved as .../E"), "{err}");
    }

    const CELL: &str = "// A place on the grid.\nstruct S_Cell at /Game/Types/S_Cell {\n    X: int\n    // How far up.\n    Y: int = 2\n    Tool: E_Tool = E_Tool.Saw\n    Next: array<S_Cell>\n}\n";

    #[test]
    fn a_struct_is_made_with_its_fields_in_order() {
        let made = &compile_together(&[TOOL, CELL]).unwrap()[1];
        assert!(made.contains(r#"{"op":"struct","path":"/Game/Types/S_Cell","source":"test.lm","fields":[{"name":"X","type":"int"},{"name":"Y","type":"int","default":"2","description":"How far up."},{"name":"Tool","type":"enum:/Game/Types/E_Tool.E_Tool","default":"Saw"},{"name":"Next","type":"array<struct:/Game/Types/S_Cell.S_Cell>"}],"description":"A place on the grid."}"#), "{made}");
    }

    #[test]
    fn a_blueprint_makes_reads_and_sets_a_struct_declared_beside_it() {
        let src = format!("{HEADER}var C: S_Cell = S_Cell(X: 1)\nfn F() -> int {{\n C.Y = 3\n let d = S_Cell(X: 4, Y: C.Y)\n return d.X + C.Y\n}}\n");
        compile_together(&[TOOL, CELL, &src]).unwrap();
        let err = compile_together(&[TOOL, CELL, &format!("{HEADER}fn F(c: S_Cell) -> int {{\n return c.Z\n}}\n")]).unwrap_err();
        assert!(err.to_string().contains("Z"), "{err}");
    }

    #[test]
    fn a_struct_has_fields_each_once_and_never_itself() {
        for (fields, want) in [
            ("", "has no fields"),
            ("    A: int\n    A: float\n", "A is declared twice"),
            ("    A: S\n", "would hold itself"),
            ("    A: int = \"x\"\n", "int literal"),
        ] {
            let err = compile_together(&[&format!("struct S at /Game/S {{\n{fields}}}\n")]).unwrap_err();
            assert!(err.to_string().contains(want), "{fields}: {err}");
        }
    }

    const USABLE: &str = "// Something that can be used.\ninterface BPI_Usable at /Game/Types/BPI_Usable {\n    // Uses it.\n    fn Use(by: Actor, times: int)\n    fn CanUse(by: Actor) -> bool\n}\n";

    #[test]
    fn an_interface_is_built_with_a_function_for_each_it_declares() {
        let made = &compile_together(&[USABLE]).unwrap()[0];
        assert!(made.contains(r#"{"op":"blueprint","path":"/Game/Types/BPI_Usable","parent":"/Script/CoreUObject.Interface","source":"test.lm"}"#), "{made}");
        assert!(made.contains(r#"{"op":"function","name":"Use","params":[{"name":"by","type":"object:/Script/Engine.Actor"},{"name":"times","type":"int"}],"description":"Uses it."}"#), "{made}");
        assert!(made.contains(r#"{"op":"function","name":"CanUse","params":[{"name":"by","type":"object:/Script/Engine.Actor"}],"returns":[{"name":"ReturnValue","type":"bool"}]}"#), "{made}");
    }

    #[test]
    fn a_blueprint_implements_and_calls_an_interface_declared_beside_it() {
        let src = format!("{HEADER}implements BPI_Usable\nvar Uses: int = 0\non Use(by: Actor, times: int) {{\n Uses = Uses + times\n}}\non CanUse(by: Actor) -> bool {{\n return Uses < 3\n}}\nfn Try(thing: Object) -> bool {{\n let u = cast(thing, BPI_Usable)\n if u != none {{\n  u.Use(self, 1)\n }}\n return u != none\n}}\n");
        compile_together(&[USABLE, &src]).unwrap();
        let err = compile_together(&[USABLE, &format!("{HEADER}implements BPI_Usable\non Use(by: Actor) {{\n}}\n")]).unwrap_err();
        assert!(err.to_string().contains("times"), "{err}");
    }

    #[test]
    fn an_interface_declares_functions_each_once_without_bodies() {
        for (functions, want) in [
            ("", "has no functions"),
            ("    fn A()\n    fn A(x: int)\n", "A is declared twice"),
            ("    fn A(x: ref int)\n", "is a ref"),
            ("    fn A(x: int, x: float)\n", "two parameters called x"),
        ] {
            let err = compile_together(&[&format!("interface I at /Game/I {{\n{functions}}}\n")]).unwrap_err();
            assert!(err.to_string().contains(want), "{functions}: {err}");
        }
        let err = parse_source("interface I at /Game/I {\n    fn A() {\n    }\n}\n").unwrap_err();
        assert!(err.to_string().contains("has no body"), "{err}");
    }

    #[test]
    fn a_count_runs_from_its_first_number_to_its_last_both_included() {
        let doc = compile(&format!("{HEADER}fn F(n: int) -> int {{\n let total = 0\n for i in 1..n - 1 {{\n  if i == 3 {{\n   continue\n  }}\n  for j in 0..i {{\n   total = total + j\n  }}\n }}\n return total\n}}\n")).unwrap();
        let json = doc.to_json();
        assert!(json.contains("LessEqual_IntInt"), "{json}");
        // The last number is not kept: it is read again before every pass.
        assert!(!doc.ops.iter().any(|op| matches!(op, Op::Function { locals, .. } if locals.iter().any(|l| l.name.starts_with("Loom_Last")))));
    }

    #[test]
    fn a_counts_last_number_is_read_before_every_pass() {
        // A call in `last` runs again before each pass, so the loop goes back to it, not to the
        // comparison.
        let doc = compile(&format!("{HEADER}var Items: array<int>\nfn Grow() -> int {{\n    Items.Add(1)\n    return Items.Length()\n}}\nfn F() {{\n    for i in 0..Grow() {{\n        if i > 5 {{\n            break\n        }}\n    }}\n}}\n")).unwrap();
        let grow = doc.ops.iter().find_map(|op| match op {
            Op::Node { id, graph, kind: NodeKind::Call { function } } if graph == "F" && function.ends_with(":Grow") => Some(id.clone()),
            _ => None,
        });
        let grow = grow.expect("F calls Grow");
        let into_grow = doc.ops.iter().filter(|op| matches!(op, Op::Link { to, .. } if to.node == grow && to.pin == "execute")).count();
        assert_eq!(into_grow, 2, "Grow runs before the first pass and again after each one");
    }

    #[test]
    fn a_read_only_field_of_a_struct_is_read_but_never_set() {
        // Neither a Make Struct node nor a Set Members node has a pin for a read-only field.
        let doc = compile(&format!("{HEADER}var Level: PerPlatformInt\nfn F() -> int {{\n return Level.Default\n}}\n"));
        doc.unwrap();
        let err = compile(&format!("{HEADER}var Level: PerPlatformInt\nfn F() {{\n Level.Default = 2\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("PerPlatformInt's Default is read-only"), "{err}");
        let err = compile(&format!("{HEADER}fn F() -> PerPlatformInt {{\n return PerPlatformInt(Default: 2)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("PerPlatformInt's Default is read-only"), "{err}");
    }

    #[test]
    fn a_for_over_an_array_may_name_the_index() {
        // The index is the counter the loop keeps, read where it is used, and cannot be set.
        let doc = compile(&format!("{HEADER}var Items: array<int>\nfn F() -> int {{\n let total = 0\n for item, i in Items {{\n  total = total + item * i\n }}\n return total\n}}\n")).unwrap();
        let counters = doc.ops.iter().filter(|op| matches!(op, Op::Node { graph, kind: NodeKind::Set { var, .. }, .. } if graph == "F" && var.contains("Index"))).count();
        assert_eq!(counters, 2, "the counter is set to 0 and then stepped, and nothing else sets it");
        let err = compile(&format!("{HEADER}var Items: array<int>\nfn F() {{\n for item, i in Items {{\n  i = 1\n }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("i is a loop's item or count"), "{err}");
        let err = compile(&format!("{HEADER}var Items: array<int>\nfn F() {{\n for a, b, c in Items {{\n }}\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("expected 'in'"), "{err}");
    }

    #[test]
    fn a_count_is_of_whole_numbers_named_once() {
        for (count, want) in [
            ("for i in 0..2.5 {\n }", "a count's last number"),
            ("for i, j in 0..2 {\n }", "names one number"),
            ("let i = 0\n for i in 0..2 {\n }", "i is already declared"),
            ("for i in 0..2 {\n  i = 1\n }", "i is a loop's item or count"),
        ] {
            let err = compile(&format!("{HEADER}fn F() {{\n {count}\n}}\n")).unwrap_err();
            assert!(err.to_string().contains(want), "{count}: {err}");
        }
    }

    #[test]
    fn a_compound_assignment_reads_its_target_then_writes_it() {
        compile(&format!("{HEADER}var Count: int = 0\nvar Name: string\nvar Flags: int = 0\nfn F(x: int) {{\n let total = 10\n total -= x\n Count += total\n Count *= 2\n Count /= 3\n Count %= 4\n Flags |= 4\n Flags &= 6\n Name += \"!\"\n}}\n")).unwrap();
        let err = compile(&format!("{HEADER}var Count: int = 0\nfn F() {{\n Count += 1.5\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("mixes types"), "{err}");
        let err = compile(&format!("{HEADER}fn F() {{\n F() += 1\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("only a variable or a member can be assigned"), "{err}");
    }

    #[test]
    fn a_compound_assignment_picks_out_its_target_once() {
        let doc = compile(&format!("{HEADER}var Items: array<int>\nvar Next: int = 0\nfn Pick() -> int {{\n Next += 1\n return Next\n}}\nfn F() {{\n Items[Pick()] += 10\n}}\n")).unwrap();
        let picks = doc.ops.iter().filter(|op| matches!(op, Op::Node { graph, kind: NodeKind::Call { function }, .. } if graph == "F" && function == "self:Pick")).count();
        assert_eq!(picks, 1);
    }

    #[test]
    fn a_call_may_leave_out_parameters_with_defaults() {
        let doc = compile(&format!("{HEADER}fn Hit(amount: int, times: int = 2, label: string = \"hit\") -> int {{\n return amount * times\n}}\nfn F() -> int {{\n return Hit(1) + Hit(1, 3) + Hit(1, 3, \"x\")\n}}\n")).unwrap();
        let hit = doc.ops.iter().find_map(|op| match op { Op::Function { name, params, .. } if name == "Hit" => Some(params.clone()), _ => None }).unwrap();
        assert_eq!(hit.iter().map(|p| p.default.as_deref()).collect::<Vec<_>>(), [None, Some("2"), Some("hit")]);
        let written: Vec<&str> = doc.ops.iter().filter_map(|op| match op { Op::Default { pin, value } if pin.pin == "times" => Some(value.as_str()), _ => None }).collect();
        assert!(written.contains(&"2"), "{written:?}");
        let err = compile(&format!("{HEADER}fn Hit(amount: int, times: int = 2) {{\n}}\nfn F() {{\n Hit()\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("1 of them are required"), "{err}");
    }

    #[test]
    fn another_blueprint_may_leave_out_a_parameter_with_a_default() {
        let lib = "blueprint BPL_Hits : BlueprintFunctionLibrary at /Game/BPL_Hits\nfn Hit(amount: int, times: int = 2) -> int {\n return amount * times\n}\n";
        compile_together(&[lib, &format!("{HEADER}fn F() -> int {{\n return BPL_Hits.Hit(4)\n}}\n")]).unwrap();
    }

    #[test]
    fn only_a_functions_last_parameters_have_defaults_written_out() {
        for (decl, want) in [
            ("fn F(a: int = 1, b: int) {\n}", "b comes after a parameter with a default"),
            ("fn F(a: ref int = 1) {\n}", "a is a ref"),
            ("dispatcher D(a: int = 1)", "only a function's parameters have defaults"),
            ("on ReceiveTick(dt: float = 1.0) {\n}", "only a function's parameters have defaults"),
            ("fn F(a: int = 1 + 1) {\n}", "must be written out"),
        ] {
            let err = compile(&format!("{HEADER}{decl}\n")).unwrap_err();
            assert!(err.to_string().contains(want), "{decl}: {err}");
        }
    }

    #[test]
    fn a_pure_function_is_a_node_off_the_chain() {
        let doc = compile(&format!("{HEADER}var Size: int = 2\nprivate pure fn Area() -> int {{\n let total = 0\n for i in 1..Size {{\n  total += Size\n }}\n return total\n}}\nfn F() -> int {{\n return Area() + Area()\n}}\n")).unwrap();
        let area = doc.ops.iter().find_map(|op| match op { Op::Function { name, pure, access, .. } if name == "Area" => Some((*pure, access.clone())), _ => None });
        assert_eq!(area, Some((true, Some("private".to_string()))));
        let chained = doc.ops.iter().any(|op| matches!(op, Op::Link { to, .. } if to.node.contains("pure") && to.pin == "execute"));
        assert!(!chained);
        let err = compile(&format!("{HEADER}pure fn Area() -> int {{\n return 1\n}}\nfn F() {{\n Area()\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("on its own it does nothing"), "{err}");
    }

    #[test]
    fn a_pure_function_changes_nothing_outside_it_unless_it_is_pure_mut() {
        let members = "var Count: int = 0\nvar Items: array<int>\nvar Where: Vector2D\ndispatcher Changed()\nfn Bump() {\n}\n";
        for (body, want) in [
            ("Count = 1", "this assigns what is outside it"),
            ("Count += 1", "this assigns what is outside it"),
            ("Where.X = 1.0", "this assigns what is outside it"),
            ("Items[0] = 1", "this assigns what is outside it"),
            ("Items.Add(1)", "Add changes what is outside it"),
            ("Bump()", "it calls Bump, which is not pure"),
            ("Changed.broadcast()", "it broadcasts Changed"),
            ("K2_DestroyActor()", "which is not pure"),
        ] {
            let err = compile(&format!("{HEADER}{members}pure fn F() -> int {{\n {body}\n return 1\n}}\n")).unwrap_err();
            assert!(err.to_string().contains(want), "{body}: {err}");
            compile(&format!("{HEADER}{members}pure mut fn F() -> int {{\n {body}\n return 1\n}}\n")).unwrap();
        }
        // What the function holds itself, it may change.
        compile(&format!("{HEADER}pure fn F(n: int) -> int {{\n let xs: array<int> = []\n xs.Add(n)\n xs[0] += 1\n let w = Vector2D(X: 1.0, Y: 2.0)\n w.X = 3.0\n return xs[0]\n}}\n")).unwrap();
    }

    #[test]
    fn a_private_or_protected_function_is_called_only_where_blueprint_allows() {
        let base = "blueprint BP_Base : Actor at /Game/BP_Base\nprivate fn Secret() {\n}\nprotected fn Family() {\n}\nfn Open() {\n Secret()\n Family()\n}\n";
        let child = "blueprint BP_Child : BP_Base at /Game/BP_Child\nfn F() {\n Family()\n}\n";
        compile_together(&[base, child]).unwrap();
        let err = compile_together(&[base, "blueprint BP_Child : BP_Base at /Game/BP_Child\nfn F() {\n Secret()\n}\n"]).unwrap_err();
        assert!(err.to_string().contains("Secret is private to BP_Base"), "{err}");
        let err = compile_together(&[base, &format!("{HEADER}fn F(b: BP_Base) {{\n b.Family()\n}}\n")]).unwrap_err();
        assert!(err.to_string().contains("Family is protected"), "{err}");
        compile_together(&[base, &format!("{HEADER}fn F(b: BP_Base) {{\n b.Open()\n}}\n")]).unwrap();
        let err = parse("blueprint B : Actor at /Game/B\npure private fn F() {\n}\n").unwrap_err();
        assert!(err.to_string().contains("write the access first"), "{err}");
    }

    #[test]
    fn a_library_calls_its_own_private_and_protected_functions_by_either_name() {
        let library = "blueprint BPL_A : BlueprintFunctionLibrary at /Game/BPL_A\nprivate fn Secret() {\n}\nprotected fn Guarded() {\n}\n\
            fn Open() {\n Secret()\n Guarded()\n BPL_A.Secret()\n BPL_A.Guarded()\n}\n";
        compile_together(&[library]).unwrap();
        let err = compile_together(&[library, &format!("{HEADER}fn F() {{\n BPL_A.Guarded()\n}}\n")]).unwrap_err();
        assert!(err.to_string().contains("Guarded is protected, so only BPL_A and"), "{err}");
        let err = compile_together(&[library, &format!("{HEADER}fn F() {{\n BPL_A.Secret()\n}}\n")]).unwrap_err();
        assert!(err.to_string().ends_with("Secret is private to BPL_A"), "{err}");
    }

    #[test]
    fn an_engine_functions_access_does_not_limit_its_callers() {
        let db = TypeDb::parse(crate::fixtures::engine()).unwrap();
        let src = format!("{HEADER}fn F(w: UserWidget) {{\n w.RegisterInputComponent()\n}}\n");
        lower(&parse(&src).unwrap(), "test.lm", &db).unwrap();
    }

    #[test]
    fn a_pure_function_cannot_be_bound() {
        let db = TypeDb::parse(crate::fixtures::engine()).unwrap();
        let bind = |qualifier: &str| {
            let src = format!("{HEADER}component Sound: AudioComponent\n{qualifier}fn Finished() {{\n}}\nfn F() {{\n Sound.OnAudioFinished.bind(Finished)\n}}\n");
            lower(&parse(&src).unwrap(), "test.lm", &db)
        };
        bind("").unwrap();
        let err = bind("pure ").unwrap_err();
        assert!(err.to_string().contains("Finished is pure"), "{err}");
    }

    /// The delegate nodes of a document, each with the function it names.
    fn delegates(doc: &Document) -> Vec<(String, String)> {
        doc.ops
            .iter()
            .filter_map(|op| match op {
                Op::Node { id, kind: NodeKind::Delegate { function }, .. } => Some((id.clone(), function.clone())),
                _ => None,
            })
            .collect()
    }

    const TIMER: &str = "KismetSystemLibrary.K2_SetTimerDelegate";

    #[test]
    fn a_function_named_where_a_delegate_is_expected_is_a_delegate_to_it() {
        let src = format!("{HEADER}fn Tick() {{\n}}\nfn F(other: Actor) {{\n {TIMER}(Tick, 0.5, true)\n {TIMER}(K2_DestroyActor, 1, false)\n {TIMER}(other.K2_DestroyActor, 1, false)\n}}\n");
        let doc = compile(&src).unwrap();
        let found = delegates(&doc);
        let functions: Vec<&str> = found.iter().map(|(_, f)| f.as_str()).collect();
        assert_eq!(functions, ["Tick", "K2_DestroyActor", "K2_DestroyActor"]);
        for (id, _) in &found {
            let fed = doc.ops.iter().any(|op| matches!(op, Op::Link { from, to } if from.node == *id && from.pin == "OutputDelegate" && to.pin == "Delegate"));
            assert!(fed, "{id} feeds the call's Delegate");
        }
        // Only another object's function has the object wired to the node.
        assert!(links_into(&doc, &found[0].0, "self").is_empty());
        assert!(links_into(&doc, &found[1].0, "self").is_empty());
        assert_eq!(links_into(&doc, &found[2].0, "self"), ["F/entry.other"]);
    }

    #[test]
    fn a_function_bound_to_a_delegate_takes_exactly_what_it_passes() {
        let wrong = |decl: &str, arg: &str| {
            compile(&format!("{HEADER}{decl}\nfn F(other: Actor) {{\n {TIMER}({arg}, 1, false)\n}}\n")).unwrap_err().to_string()
        };
        let err = wrong("fn Tick(x: int) {\n}", "Tick");
        assert!(err.contains("Tick is fn(x: int), and a function bound to TimerDynamicDelegate must be fn()"), "{err}");
        assert!(wrong("fn Tick() -> int {\n return 1\n}", "Tick").contains("Tick is fn() -> int"));
        // An output is part of what a function takes, so one with an output fits no signature
        // without it.
        let err = wrong("fn Tick(out done: bool) {\n done = true\n}", "Tick");
        assert!(err.contains("Tick is fn(out done: bool)"), "{err}");
        assert!(wrong("pure fn Tick() -> int {\n return 1\n}", "Tick").contains("Tick is pure"));
        assert!(wrong("", "other.K2_GetActorLocation").contains("K2_GetActorLocation is pure"));
        assert!(wrong("", "KismetSystemLibrary.PrintString").contains("PrintString is static"));
        assert!(wrong("", "Nope").contains("no variable Nope"));
        assert!(wrong("", "other.Nope").contains("Actor has no Nope"));
        // A const reference is matched by a ref parameter, and only by one.
        let judge = |params: &str| compile(&format!("{HEADER}fn Rate({params}) -> bool {{\n return true\n}}\nfn F() {{\n Judges.Judge(Rate)\n}}\n"));
        judge("score: int, where: ref Vector2D").unwrap();
        let err = judge("score: int, where: Vector2D").unwrap_err().to_string();
        assert!(err.contains("must be fn(Score: int, Where: ref Vector2D) -> bool"), "{err}");
        let err = compile("blueprint L : BlueprintFunctionLibrary at /Game/L\nfn Tick() {\n}\nfn F() {\n KismetSystemLibrary.K2_SetTimerDelegate(Tick, 1, false)\n}\n").unwrap_err();
        assert!(err.to_string().contains("belongs to no object"), "{err}");
    }

    #[test]
    fn a_delegate_is_a_functions_parameter_or_return_value_and_nothing_else() {
        let src = format!(
            "{HEADER}fn Tick() {{\n}}\nfn Every(seconds: float, handler: TimerDynamicDelegate) {{\n {TIMER}(handler, seconds, true)\n}}\nfn Pick() -> TimerDynamicDelegate {{\n return Tick\n}}\nfn F() {{\n Every(1, Tick)\n Every(2, Pick())\n}}\n"
        );
        let doc = compile(&src).unwrap();
        let delegate = crate::ops::Type::Delegate("/Script/Engine.TimerDynamicDelegate__DelegateSignature".into());
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Function { name, params, .. } if name == "Every" && params[1].ty == delegate)));
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Function { name, returns, .. } if name == "Pick" && returns[0].ty == delegate)));
        let wrong = |items: &str| compile(&format!("{HEADER}fn Tick() {{\n}}\n{items}\n")).unwrap_err().to_string();
        for (items, why) in [
            ("var D: TimerDynamicDelegate", "no delegate variables"),
            ("var D: array<TimerDynamicDelegate>", "cannot hold a delegate"),
            ("fn F() {\n let d: TimerDynamicDelegate = Tick\n}", "no delegate variables"),
            ("fn F(d: TimerDynamicDelegate) {\n d()\n}", "cannot call a delegate"),
            ("fn F() {\n let d = Tick\n}", "Tick is a function"),
            ("fn F(a: TimerDynamicDelegate, b: TimerDynamicDelegate) {\n KismetSystemLibrary.K2_SetTimerDelegate(if true { a } else { b }, 1, false)\n}", "cannot choose"),
        ] {
            let err = wrong(items);
            assert!(err.contains(why), "{items}: {err}");
        }
        let err = parse_source("struct S at /Game/S {\n    D: TimerDynamicDelegate\n}\n").and_then(|s| lower_source(&s, "s.lm", &db())).unwrap_err();
        assert!(err.to_string().contains("refuses a delegate field"), "{err}");
    }

    #[test]
    fn a_dispatcher_passes_a_delegate_to_the_functions_bound_to_it() {
        let src = format!(
            "{HEADER}dispatcher Later(seconds: float, done: TimerDynamicDelegate)\nfn Tick() {{\n}}\nfn Wait(seconds: float, done: TimerDynamicDelegate) {{\n {TIMER}(done, seconds, false)\n}}\nfn F() {{\n Later.bind(Wait)\n Later.broadcast(0.5, Tick)\n}}\n"
        );
        let doc = compile(&src).unwrap();
        let delegate = crate::ops::Type::Delegate("/Script/Engine.TimerDynamicDelegate__DelegateSignature".into());
        assert!(doc.ops.iter().any(|op| matches!(op, Op::Dispatcher { name, params, .. } if name == "Later" && params[1].ty == delegate)));
        // What is bound takes a delegate of the same signature.
        let err = compile(&format!("{HEADER}dispatcher Later(done: TimerDynamicDelegate)\nfn Wait(done: OnDouble) {{\n}}\nfn F() {{\n Later.bind(Wait)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("Wait is fn(done: OnDouble)"), "{err}");
    }

    #[test]
    fn an_interfaces_function_takes_and_returns_a_delegate() {
        let timed = "interface BPI_Timed at /Game/Types/BPI_Timed {\n    fn Delay(seconds: float, done: TimerDynamicDelegate)\n    fn Chosen() -> TimerDynamicDelegate\n}\n";
        let made = &compile_together(&[timed]).unwrap()[0];
        assert!(made.contains(r#"{"name":"done","type":"delegate:/Script/Engine.TimerDynamicDelegate__DelegateSignature"}"#), "{made}");
        assert!(made.contains(r#""returns":[{"name":"ReturnValue","type":"delegate:/Script/Engine.TimerDynamicDelegate__DelegateSignature"}]"#), "{made}");
        let src = format!(
            "{HEADER}implements BPI_Timed\nfn Tick() {{\n}}\non Delay(seconds: float, done: TimerDynamicDelegate) {{\n {TIMER}(done, seconds, false)\n}}\non Chosen() -> TimerDynamicDelegate {{\n return Tick\n}}\nfn F() {{\n let timed = cast(self, BPI_Timed)\n timed.Delay(1, Tick)\n {TIMER}(timed.Chosen(), 1, false)\n}}\n"
        );
        compile_together(&[timed, &src]).unwrap();
        let err = compile_together(&[timed, &format!("{HEADER}implements BPI_Timed\non Delay(seconds: float, done: OnDouble) {{\n}}\n")]).unwrap_err();
        assert!(err.to_string().contains("Delay's done is TimerDynamicDelegate, not OnDouble"), "{err}");
    }

    #[test]
    fn a_function_bound_to_a_delegate_takes_its_floats_as_wide_as_it_passes_them() {
        // This Blueprint's own function takes a 32-bit float where the engine passes one.
        let src = format!("{HEADER}fn Got(value: float) {{\n}}\nfn Kept(value: float) {{\n}}\nfn F() {{\n Judges.Watch(Got)\n Judges.Hold(Kept)\n Got(1.5)\n}}\n");
        let doc = compile(&src).unwrap();
        let param = |function: &str| {
            doc.ops.iter().find_map(|op| match op {
                Op::Function { name, params, .. } if name == function => Some(params[0].ty.clone()),
                _ => None,
            })
        };
        assert_eq!(param("Got"), Some(crate::ops::Type::Float32));
        assert_eq!(param("Kept"), Some(crate::ops::Type::Float));
        // It can't take both widths.
        let wrong = |items: &str, body: &str| {
            compile(&format!("{HEADER}fn Got(value: float) {{\n}}\n{items}fn F(other: Actor, copies: Copies) {{\n {body}\n}}\n")).unwrap_err().to_string()
        };
        let err = wrong("", "Judges.Watch(Got)\n Judges.Hold(Got)");
        assert!(err.contains("Got is bound to OnDouble, which passes Value as a 64-bit float, and at 5:15 to OnSingle, which passes a 32-bit one"), "{err}");
        let err = wrong("dispatcher Changed(value: float)\n", "Changed.bind(Got)\n Judges.Watch(Got)");
        assert!(err.contains("and at 6:15 to Changed, which passes a 64-bit one"), "{err}");
        // A function loom can't make take a 32-bit float must take one already.
        compile(&format!("{HEADER}fn F(other: Actor) {{\n Judges.Watch(other.SetLifeSpan)\n}}\n")).unwrap();
        let err = wrong("", "Judges.Watch(copies.Take)");
        assert!(err.contains("Take takes Amount as a 64-bit float, and OnSingle passes a 32-bit one"), "{err}");
    }

    #[test]
    fn a_dispatcher_binds_unbinds_and_unbinds_all() {
        let src = format!(
            "{HEADER}dispatcher Done(score: int)\nfn Scored(score: int) {{\n}}\nfn Gone(actor: Actor) {{\n}}\nfn F(other: Actor) {{\n Done.bind(Scored)\n self.Done.unbind(Scored)\n Done.unbind_all()\n other.OnDestroyed.bind(Gone)\n}}\n"
        );
        let doc = compile(&src).unwrap();
        let kinds: Vec<&NodeKind> = doc
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::Node { kind: kind @ (NodeKind::AddDelegate { .. } | NodeKind::RemoveDelegate { .. } | NodeKind::ClearDelegate { .. }), .. } => Some(kind),
                _ => None,
            })
            .collect();
        assert_eq!(kinds[0], &NodeKind::AddDelegate { class: None, dispatcher: "Done".into() });
        assert_eq!(kinds[1], &NodeKind::RemoveDelegate { class: None, dispatcher: "Done".into() });
        assert_eq!(kinds[2], &NodeKind::ClearDelegate { class: None, dispatcher: "Done".into() });
        assert_eq!(kinds[3], &NodeKind::AddDelegate { class: Some("/Script/Engine.Actor".into()), dispatcher: "OnDestroyed".into() });
        assert_eq!(delegates(&doc).len(), 3);
        let wrong = |body: &str| compile(&format!("{HEADER}dispatcher Done(score: int)\nfn Scored(score: int) {{\n}}\nfn F() {{\n {body}\n}}\n")).unwrap_err().to_string();
        assert!(wrong("Done.bind(F)").contains("F is fn(), and a function bound to Done must be fn(score: int)"));
        assert!(wrong("Done.unbind()").contains("unbind() takes a function"));
        assert!(wrong("Done.unbind_all(Scored)").contains("unbind_all() takes nothing"));
        assert!(wrong("self.Nope.bind(Scored)").contains("has no dispatcher Nope"));
        let pure = compile(&format!("{HEADER}dispatcher Done(score: int)\nfn Scored(score: int) {{\n}}\npure fn F() -> int {{\n Done.unbind_all()\n return 1\n}}\n")).unwrap_err();
        assert!(pure.to_string().contains("changes what Done calls"), "{pure}");
    }

    const LOOKUP: &str = "fn Lookup(key: string, out found: bool, out value: int) {\n found = key == \"a\"\n if found {\n value = 1\n return\n }\n value = value + 10\n}\n";

    #[test]
    fn an_out_parameter_is_an_output_fed_by_every_return_and_the_end_of_the_body() {
        let doc = compile(&format!("{HEADER}{LOOKUP}")).unwrap();
        let (returns, locals) = doc
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Function { name, returns, locals, params, .. } if name == "Lookup" => {
                    assert_eq!(params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["key"]);
                    Some((returns.clone(), locals.clone()))
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(returns.iter().map(|p| (p.name.as_str(), p.ty.to_string())).collect::<Vec<_>>(), [("found", "bool".into()), ("value", "int".into())]);
        assert_eq!(locals.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["Loom_Out_found", "Loom_Out_value"]);
        // The `return` and the end of the body each feed both outputs to a result node of their own.
        let results: Vec<&str> = doc.ops.iter().filter_map(|op| match op { Op::Node { id, kind: NodeKind::Result, .. } => Some(id.as_str()), _ => None }).collect();
        assert_eq!(results.len(), 2, "{results:?}");
        for result in results {
            for output in ["found", "value"] {
                let from = links_into(&doc, result, output);
                assert_eq!(from.len(), 1, "{result}.{output}: {from:?}");
                assert!(from[0].ends_with(&format!(".Loom_Out_{output}")), "{result}.{output} comes from {}", from[0]);
            }
        }
        // The outputs are assigned and read as the locals that hold them.
        assert!(doc.to_json().contains(r#""kind":"set","var":"Loom_Out_value","scope":"local""#));
        assert!(doc.to_json().contains(r#""kind":"get","var":"Loom_Out_value","scope":"local""#));
    }

    #[test]
    fn a_call_with_outputs_gives_a_record_read_by_name() {
        let src = format!("{HEADER}{LOOKUP}fn Use() -> int {{\n let r = Lookup(\"a\")\n if r.found {{\n return r.value\n }}\n Lookup(\"b\")\n return 0\n}}\n");
        let doc = compile(&src).unwrap();
        assert_eq!(links_into(&doc, "Use/if2", "Condition"), ["Use/call1.found"]);
        assert_eq!(links_into(&doc, "Use/return3", "ReturnValue"), ["Use/call1.value"]);
        // With a return value too, that is ReturnValue, and a pure function's outputs are read off the chain.
        let src = format!("{HEADER}pure fn Divide(n: int, by: int, out remainder: int) -> int {{\n remainder = n % by\n return n / by\n}}\nfn F() -> int {{\n let d = Divide(7, 2)\n return d.ReturnValue + d.remainder\n}}\n");
        let doc = compile(&src).unwrap();
        assert_eq!(links_into(&doc, "F/pure2", "A"), ["F/pure1.ReturnValue"]);
        assert_eq!(links_into(&doc, "F/pure2", "B"), ["F/pure1.remainder"]);
        assert!(links_into(&doc, "F/pure1", "execute").is_empty(), "a pure function's node is off the chain");
    }

    #[test]
    fn an_output_is_neither_passed_nor_taken_as_one_value() {
        let err = compile(&format!("{HEADER}{LOOKUP}fn F() {{\n Lookup(\"a\", true)\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("found and value are its outputs, which a call does not pass"), "{err}");
        let err = compile(&format!("{HEADER}{LOOKUP}fn F() -> int {{\n return Lookup(\"a\")\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("pick one by name"), "{err}");
        let err = compile(&format!("{HEADER}{LOOKUP}fn F() {{\n let x: int = Lookup(\"a\")\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("holds a call's outputs"), "{err}");
        let err = compile(&format!("{HEADER}{LOOKUP}fn F() {{\n let r = Lookup(\"a\")\n let n = r.missing\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("has no missing"), "{err}");
    }

    #[test]
    fn an_output_is_declared_once_and_is_not_a_local_or_the_return_pin() {
        let err = compile(&format!("{HEADER}fn F(out n: int) {{\n let n = 1\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("n is already declared"), "{err}");
        let err = compile(&format!("{HEADER}fn F(n: int, out n: int) {{\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("two parameters called n"), "{err}");
        let err = compile(&format!("{HEADER}fn F(out ReturnValue: int) -> int {{\n return 1\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("cannot be called ReturnValue"), "{err}");
        // Without a return value, the name is free.
        compile(&format!("{HEADER}fn F(out ReturnValue: int) {{\n ReturnValue = 1\n}}\n")).unwrap();
        // An output may come first, and a call passes the inputs in their order.
        compile(&format!("{HEADER}fn F(out n: int, a: int, b: int = 2) {{\n n = a + b\n}}\nfn G() {{\n F(1)\n let r = F(1, 3)\n let m = r.n\n}}\n")).unwrap();
    }

    #[test]
    fn only_a_function_has_outputs() {
        let err = compile(&format!("{HEADER}on ReceiveTick(out dt: float) {{\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("dt cannot be out"), "{err}");
        let src = format!("{HEADER}component Sound: AudioComponent\non Sound.OnAudioFinished(out n: int) {{\n}}\n");
        let err = lower(&parse(&src).unwrap(), "test.lm", &TypeDb::parse(crate::fixtures::engine()).unwrap()).unwrap_err();
        assert!(err.to_string().contains("n cannot be out"), "{err}");
    }

    const COUNTER: &str = "interface BPI_Counter at /Game/Types/BPI_Counter {\n    fn Count(of: string, out found: bool) -> int\n}\n";

    #[test]
    fn an_interfaces_function_gives_back_outputs_which_its_handler_sets_by_name() {
        let made = &compile_together(&[COUNTER]).unwrap()[0];
        assert!(made.contains(r#""returns":[{"name":"ReturnValue","type":"int"},{"name":"found","type":"bool"}]"#), "{made}");
        let handler = format!("{HEADER}implements BPI_Counter\non Count(of: string, out found: bool) -> int {{\n found = true\n return 1\n}}\nfn Try(thing: Object) -> bool {{\n let r = cast(thing, BPI_Counter).Count(\"a\")\n return r.found\n}}\n");
        compile_together(&[COUNTER, &handler]).unwrap();
        let unnamed = format!("{HEADER}implements BPI_Counter\non Count(of: string) -> int {{\n return 1\n}}\n");
        let err = compile_together(&[COUNTER, &unnamed]).unwrap_err();
        assert!(err.to_string().contains("write it as `out found: bool`"), "{err}");
        let renamed = format!("{HEADER}implements BPI_Counter\non Count(of: string, out ok: bool) -> int {{\n return 1\n}}\n");
        let err = compile_together(&[COUNTER, &renamed]).unwrap_err();
        assert!(err.to_string().contains("write it as `out found: bool`"), "{err}");
    }

    #[test]
    fn a_pure_function_may_assign_its_own_outputs() {
        compile(&format!("{HEADER}pure fn F(out n: int) {{\n n = 1\n n += 1\n}}\n")).unwrap();
    }

    #[test]
    fn an_event_handler_names_the_events_parameters() {
        compile(&format!("{HEADER}on ReceiveTick(dt: float) {{\n}}\n")).unwrap();
        let err = compile(&format!("{HEADER}on ReceiveTick {{\n}}\n")).unwrap_err();
        assert!(err.to_string().contains("DeltaSeconds"), "{err}");
    }
}

/// Each example's source compiled to ops.json, compared with the committed `compiled.json` beside
/// it, so a change to lowering shows up as a diff. `LOOM_BLESS=1 cargo test` rewrites them.
#[cfg(test)]
mod snapshots {
    use std::path::Path;

    use crate::syntax::parse_file;

    #[test]
    fn examples_compile_to_their_snapshots() {
        let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let bless = std::env::var_os("LOOM_BLESS").is_some();
        let mut sources = Vec::new();
        for entry in std::fs::read_dir(&examples).unwrap() {
            let dir = entry.unwrap().path();
            for source in std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()) {
                if source.extension().is_some_and(|e| e == "lm") {
                    sources.push((dir.clone(), source));
                }
            }
        }
        // The examples are one program, as they are built into one project, so one may use another.
        let files: Vec<_> = sources.iter().map(|(_, source)| parse_file(source).unwrap()).collect();
        let mut db = super::tests::db();
        super::declare(&files, &mut db).unwrap();
        let mut checked = 0;
        for ((dir, source), file) in sources.iter().zip(&files) {
            // Named with `/` as the snapshot records it, whatever the platform's separator.
            let folder = dir.file_name().unwrap().to_string_lossy();
            let name = format!("examples/{folder}/{}", source.file_name().unwrap().to_string_lossy());
            let got = super::lower_source(file, &name, &db).unwrap().to_json() + "\n";
            let snapshot = dir.join("compiled.json");
            if bless {
                std::fs::write(&snapshot, &got).unwrap();
            } else {
                let want = std::fs::read_to_string(&snapshot)
                    .unwrap_or_else(|_| panic!("no {}; run LOOM_BLESS=1 cargo test", snapshot.display()));
                assert!(
                    got == want,
                    "{name} no longer compiles to {}; run LOOM_BLESS=1 cargo test and review the diff",
                    snapshot.display()
                );
            }
            checked += 1;
        }
        assert!(checked > 0);
    }
}
