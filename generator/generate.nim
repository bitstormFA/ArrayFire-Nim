import std/json
import std/strutils
import std/sequtils
import std/options
import std/tables
import std/strformat
import std/algorithm
import std/sets
import print
from std/re import nil

var type_rename = {"array": "AFArray", "dim_t": "DimT",
                   "af_array": "AF_Array_Handle",
                   "ArrayProxy": "AFArray",
                   "cuchar": "cstring",
                   "exception": "AF_Exception",
                   "seq": "AF_Seq", "af_cfloat": "cdouble",
                   "af_cdouble": "cdouble", "cdouble": "cdouble",
                   "af_someenum_t": "SomeenumT", "cfloat": "cdouble",
                   "af_seq": "AF_Seq", "index": "IndexT",
                   ":int": "cint", ":unsigned-int": "cuint", ":char": "cstring",
                   ":unsigned-char": "cstring", 
                   ":_Bool": "bool", ":long": "clong", "CSpace": "CSpaceT",
                   ":unsigned-long": "culong", ":long-long": "clonglong",
                   ":unsigned-long-long": "culonglong", ":double": "cdouble",
                   ":float": "cdouble", ":short": "cshort", ":void": "pointer",
                   ":unsigned-short": "cushort", "size_t": "csize_t",
                   "ptr": "pointer"}.toTable()

const function_rename = {"mod": "af_mod", "alloc": "af_alloc", "var": "af_var", "block": "af_block", "=": "assign",
                         "type": "dtype", "array": "af_array", "sum": "asum", "min": "amin", "max": "amax",
                         "seq": "af_seq", "as": "af_as", "Window": "make_window", "()": "call"}.toTable()

const skip_functions = @["batchFunc", "timeit", "tile"]

const skip_enums: seq[string] = @[]

const skip_types =  @["array_proxy", "ArrayProxy"]

const operator_rename = {"array": "afa"}.toTable()

const parameter_rename = {"in": "af_in", "out": "af_out", "from": "af_from", "seq": "af_seq",
                          "method": "af_mathod", "var": "af_var", "func": "af_func",
                          "end": "af_end", "type": "af_type", "s_": "s", "": "p",
                          "ptr": "af_ptr"}.toTable()

const skip_signatures = @["operator==cdouble_cdouble", "operator/cdouble_cdouble", 
                          "operator*cdouble_cdouble", "operator+cdouble_cdouble", 
                          "operator-cdouble_cdouble", "operator!=cdouble_cdouble",
                          "evalafarray"]


type
    DefType{.pure.} = enum
        Enum, Typedef, Pointer, Other

    TypeEntry = object
        name: string
        dtype: DefType
        id: int
        content: string

    EnumField = object
        name: string
        eval: int

    EnumDef = object
        name: string
        fields: seq[EnumField]

    CType = object
        name: string
        bit_size: int

    Parameter = object
        name: string
        reference: bool
        ctype: CType

    Function = object
        name: string
        ns: string
        parameters: seq[Parameter]
        return_type: CType

    ClassDef = object
        name: string
        ns: string
        id: int
        bit_size: int
        methods: seq[Function]

proc tanslate_predefined_types(n: string): string =
    result = type_rename.getOrDefault(n, n)

proc make_type_name(n: string): string =
    if n.startsWith(":"):
        return n
    var start_name = n
    if n in type_rename:
        result = type_rename[n]
    else:
        var elements = start_name.split("_")
        if elements[0] == "af":
            elements.delete(0)
        for e in elements:
            if e == "cdouble":
                echo "hit"
            let upperName = e[0].toUpperAscii() & e[1 .. ^1]
            result = result & upperName
    type_rename[n] = result

proc analyze_type_node(node: JsonNode): Option[TypeEntry] =
    let name = node["name"].getStr()
    let type_child = node["type"]
    let tag = type_child["tag"].getStr()
    let id = type_child{"id"}.getInt(-1)
    let dtype = case tag:
        of ":enum": DefType.Enum
        of ":int": DefType.Typedef
        of ":long": DefType.Typedef
        of ":long-long": DefType.Typedef
        of ":pointer": DefType.Pointer
        else:
            DefType.Other
    if dtype == DefType.Other:
        return none(TypeEntry)
    else:
        return some(TypeEntry(name: name, dtype: dtype, id: id, content: tag))

proc childs_with_tag(node: JsonNode, tag: string): seq[JsonNode] =
    for child in node.items():
        if child{"tag"}.getStr("") == tag:
            result.add(child)

proc get_type_defs(node: JsonNode): (Table[int, TypeEntry], Table[string, TypeEntry]) =
    var enums = Table[int, TypeEntry]()
    var type_defs = Table[string, TypeEntry]()
    for child in node.childs_with_tag("typedef"):
        let td = analyze_type_node(child)
        if td.isSome():
            let td = td.get()
            case td.dtype:
                of DefType.Enum:
                    enums[td.id] = td
                of DefType.Typedef:
                    type_defs[td.name] = td
                else:
                    discard
    return (enums, type_defs)

proc get_namespaces(node: JsonNode): Table[int, string] =
    for child in node.childs_with_tag("namespace"):
        if child.hasKey("id") and child.hasKey("name"):
            let name = child["name"].getStr()
            let id = child["id"].getInt()
            result[id] = name

proc get_enums(node: JsonNode, type_defs: Table[int, TypeEntry]): seq[EnumDef] =
    for child in node.childs_with_tag("enum"):
        let id = child["id"].getInt()
        if not type_defs.hasKey(id):
            continue
        let td = type_defs[id]
        let name = tanslate_predefined_types(td.name)
        var enum_fields = newSeq[EnumField]()
        for field in child["fields"]:
            let field_name = field["name"].getStr()
            let field_val = field["value"].getInt()
            enum_fields.add(EnumField(name: field_name, eval: field_val))
        result.add(EnumDef(name: name, fields: enum_fields))

proc get_type(node: JsonNode): CType =
    var type_node = if node.hasKey("type"): node["type"] else: node
    let tag = type_node["tag"].getStr()
    let name = make_type_name(type_node{"name"}.getStr(tag))
    let bit_size = node{"bit-size"}.getInt(0)
    return CType(name: name, bit_size: 0)


proc get_parameters(node: JsonNode): seq[Parameter] =
    var reference = false
    for p in node["parameters"]:
        var name = p["name"].getStr()
        name = re.replace(name, re.re"_$", "")
        var type_node = p["type"]
        if type_node.hasKey("type"): #reference
            reference = true
            type_node = type_node["type"]
        let ctype = get_type(type_node)
        result.add(Parameter(name: name, ctype: ctype, reference: reference))


proc get_method_function(node: JsonNode, name_spaces: Table[int, string]): Function =
    let name = node["name"].getStr().strip()
    let parameters = get_parameters(node)
    let return_type = get_type(node["return-type"])
    let ns_id = node{"ns"}.getInt(-111)
    let namespace = name_spaces.getOrDefault(ns_id, "")
    result = Function(name: name, ns: namespace, parameters: parameters,
            return_type: return_type)

proc get_classes(node: JsonNode, name_spaces: Table[int, string]): seq[ClassDef] =
    for child in node.childs_with_tag("class"):
        let name = child["name"].getStr()
        let id = child["id"].getInt()
        let ns_id = child{"ns"}.getInt(-1)
        let ns = name_spaces.getOrDefault(ns_id, "")
        var methods = newSeq[Function]()
        for m in child["methods"]:
            let cmethod = get_method_function(m, name_spaces)
            methods.add(cmethod)
        let class_def = ClassDef(name: name, ns: ns, id: id, methods: methods)
        if len(methods) > 0:
            result.add(class_def)

proc af_childs(node: JsonNode): JsonNode = # find all nodes which belong to the af namespace
    var af_node = %* []
    # filter out nodes for af definition
    for e in node.items():
        if "/af/" in e{"location"}.getStr("") or e{"name"}.getStr("") == "af":
            af_node.add(e)
    return af_node

proc get_functions(node: JsonNode, name_spaces: Table[int, string]): seq[Function] =
    for child in node.childs_with_tag("function"):
        let f = get_method_function(child, name_spaces)
        result.add(f)

proc enum_entries_sort(x, y: EnumField): int =
    if (x.eval > y.eval) or (x.eval == y.eval and len(x.name) > len(y.name)): 1
    else: -1

proc sort_filter_enum_entries(s: seq[EnumField]): seq[EnumField] =
    let sorted_entries = s.sorted(enum_entries_sort)
    var already_used = newSeq[int]()
    for ee in sorted_entries:
        if ee.eval notin already_used:
            already_used.add(ee.eval)
            result.add(ee)

proc render_enum(e: EnumDef, public: bool = true): string =
    let public_flag = if public: "*" else: ""
    let name = make_type_name(e.name)
    result = &"""  {name}{public_flag} {{.pure, header : "arrayfire.h", import_cpp: "{e.name}", size: sizeof(cint).}} = enum """ & "\n"
    var items = newSeq[string]()
    for ee in sort_filter_enum_entries(e.fields):
        items.add(fmt"    {ee.name.toUpperAscii()} = {ee.eval}")
    let items_s = items.join(",\n")
    result = result & items_s & "\n"
    type_rename[e.name] = name

proc render_ctype(c: CType, is_return_type:bool = false): string =
    if is_return_type and c.name == ":void":  # has a different meeting in return type
        result = ":void"
    else:
        result =  tanslate_predefined_types(c.name)


proc render_parameter(p: Parameter): string =
    let type_name = render_ctype(p.ctype)
    let parameter_name = parameter_rename.getOrDefault(p.name, p.name)
    result = fmt"{parameter_name} : {type_name}"

proc render_parameters(ps: seq[Parameter]): string =
    var p_strings = newSeq[string]()
    for p in ps:
        p_strings.add(render_parameter(p))
    result = if len(p_strings) > 0: p_strings.join(", ") else: ""

proc skip_function(f: Function): bool =
    result = false
    if f.name in skip_functions:
        return true
    for p in f.parameters:
        if p.ctype.name.toLowerAscii == "istream" or
                p.ctype.name.toLowerAscii == "ostream":
            result = true
            break
        if p.ctype.name.contains("invalid"):
            result = true
            break

proc render_destructor(c: ClassDef): string =
    let type_name = tanslate_predefined_types(c.name)
    result = &"""proc destroy_{c.name}*(this: var {type_name}) {{.importcpp: "#.~{c.name}()", header : "arrayfire.h".}}"""

proc render_function(f: Function, public: bool = true, is_constructor: bool = false, is_method: bool = false): string =
    let public_flag = if public: "*" else: ""
    let parameter_string = render_parameters(f.parameters)
    let rt_raw = render_ctype(f.return_type, is_return_type=true)
    let rt = if rt_raw == ":void": "" else: &": {rt_raw}"
    let ns_prefix = if f.ns == "": "" else: &"{f.ns}::"

    if f.name.startsWith("operator"):
        var name = f.name.replace("operator", "").strip()
        name = operator_rename.getOrDefault(name, name)
        let nim_name = function_rename.getOrDefault(name, name)
        result = "proc `" & $nim_name & "`"
        result = result & &"{public_flag}({parameter_string}) {rt} " & &"""{{.importcpp: "(# {name} #)", header: "arrayfire.h".}}"""
    else:
        let nim_name = function_rename.getOrDefault(f.name, f.name)
        result = &"proc {nim_name}{public_flag}"
        result = result & fmt"( {parameter_string} ) {rt} "
        if is_constructor:
            result = result & &"""{{.constructor, importcpp: "af::{f.name}(@)", header: "arrayfire.h".}}"""
        elif is_method:
            result = result & &"""{{.importcpp: "{ns_prefix}{f.name}", header : "arrayfire.h".}}"""
        else:
            result = result & &"""{{.importcpp: "{ns_prefix}{f.name}(@)", header: "arrayfire.h".}}"""

proc render_class_methods(c: ClassDef): seq[string] =
    let this_ctype = CType(name: c.name, bit_size: 0)
    let this_parameter = Parameter(name: "this", reference: false,
            ctype: this_ctype)
    var already_processed = newSeq[Function]()
    for m in c.methods:
        if skip_function(m):
            continue
        if m.name.startsWith("~"):
            result.add(render_destructor(c))
        else:
            if m.name.contains("[]"): continue  # index will be handled in handwritten code
            let is_constructor = if m.name == c.name: true else: false
            var return_type = if is_constructor: CType(name: c.name,
                    bit_size: 0) else: m.return_type
            if m.name.endsWith("=") and not m.name.endsWith("=="):
                return_type = CType(name: ":void", bit_size:0)

            let tt = @[this_parameter]
            var new_params = if not is_constructor: tt & m.parameters else: m.parameters
            var f = Function(name: m.name, ns: m.ns, parameters: new_params,
                    return_type: return_type)
            if f notin already_processed:
                result.add(render_function(f, is_constructor = is_constructor, ismethod=true))
                already_processed.add(f)

proc render_type(c: ClassDef): string =
    let nim_type = tanslate_predefined_types(c.name)
    let imp = if c.name == "AFArray_proxy": "af::array" else: "af"
    return &"""  {nim_type}* {{.final, header : "arrayfire.h", importcpp: "{imp}::{c.name}".}} = object"""

proc function_signature(f: Function): string = 
    var param_types = newSeq[string]()
    for p in f.parameters:
        param_types.add(render_ctype(p.ctype).toLowerAscii())
    result = f.name.toLowerAscii() & param_types.join("_") 

proc generate(json_def_file: string, outfile_name: string) =
    let outfile = open(outfile_name, fmWrite)
    defer: outfile.close()

    var af_node = af_childs(parseFile(json_def_file))

    let (enums_types, type_defs) = get_type_defs(af_node)
    let namespaces = get_namespaces(af_node)
    let enums = get_enums(af_node, enums_types)
    let cclasses = get_classes(af_node, name_spaces)
    let functions = get_functions(af_node, name_spaces)

    outfile.write("""
when defined(Windows): 
  from os import nil 
  const AF_INCLUDE_PATH = os.joinPath(os.getEnv("AF_PATH"), "include") 
  const AF_LIB_PATH =  os.joinPath(os.getEnv("AF_PATH"), "lib")
  {.passC: "-D __FUNCSIG__ -std=c++11" & " -I " & AF_INCLUDE_PATH.}
  {.passL: "-lopengl32 -laf" & " -L " & AF_LIB_PATH.}
elif defined(Linux):
  {.passC: "-std=c++11".}
  {.passL: "-lGL -laf".}
elif defined(MacOsX):
  from os import nil
  const AF_INCLUDE_PATH = os.joinPath(os.getEnv("AF_PATH"), "include")
  const AF_LIB_PATH = os.joinPath(os.getEnv("AF_PATH"), "lib")
  {.passC: "-std=c++11" & " -I " & AF_INCLUDE_PATH.}
  {.passL: "-laf" & " -L " & AF_LIB_PATH.}
when sizeof(int) == 8:
  type DimT* = clonglong
else:
  type DimT* = cint 

type
  AF_Array_Handle* = distinct pointer
""")

    outfile.write("\n#region Types \n\n")
    outfile.write("type\n")
    for c in cclasses:
        if c.name in skip_types:
            continue
        outfile.write(render_type(c))
        outfile.write("\n")
    outfile.write("\n#endregion\n\n")


    outfile.write("#region Enums\n\n")
    outfile.write("type\n")
    for e in enums:
        if e.name in skip_enums: continue
        outfile.write(render_enum(e))
        outfile.write("\n")
    outfile.write("#endregion\n\n")


    outfile.write("\n#region Functions\n\n")
    var process_functions = newSeq[string]()
    for f in functions:
        if skip_function(f):
            continue
        let sig = function_signature(f)
        if sig in skip_signatures:
            continue
        if sig notin process_functions:
            process_functions.add(sig)
            outfile.write(render_function(f))
            outfile.write("\n")
    outfile.write("#endregion\n\n ")


    outfile.write("\n#region Methods\n")
    for c in cclasses:
        if c.name in skip_types:
            continue
        let method_renders = deduplicate(render_class_methods(c))
        outfile.write(method_renders.join("\n"))
        outfile.write("\n")
    outfile.write("\n#endregion\n\n")


generate("generator/arrayfire.json", "src/ArrayFire_Nim/raw.nim")
