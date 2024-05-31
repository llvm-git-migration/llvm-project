// RUN: mkdir -p %T/clang-doc/build
// RUN: mkdir -p %T/clang-doc/include
// RUN: mkdir -p %T/clang-doc/src
// RUN: mkdir -p %T/clang-doc/docs
// RUN: sed 's|$test_dir|%/T/clang-doc|g' %S/Inputs/clang-doc-project1/database_template.json > %T/clang-doc/build/compile_commands.json
// RUN: cp %S/Inputs/clang-doc-project1/*.h  %T/clang-doc/include
// RUN: cp %S/Inputs/clang-doc-project1/*.cpp %T/clang-doc/src
// RUN: cd %T/clang-doc/build
// RUN: clang-doc --format=html --repository=github.com --executor=all-TUs --output=%T/clang-doc/docs ./compile_commands.json
// RUN: FileCheck -input-file=%T/clang-doc/docs/index_json.js -check-prefix=CHECK-JSON-INDEX %s
// RUN: FileCheck -input-file=%T/clang-doc/docs/GlobalNamespace/Shape.html -check-prefix=CHECK-HTML-SHAPE %s
// RUN: FileCheck -input-file=%T/clang-doc/docs/GlobalNamespace/Calculator.html -check-prefix=CHECK-HTML-CALC %s
// RUN: FileCheck -input-file=%T/clang-doc/docs/GlobalNamespace/Rectangle.html -check-prefix=CHECK-HTML-RECTANGLE %s
// RUN: FileCheck -input-file=%T/clang-doc/docs/GlobalNamespace/Circle.html -check-prefix=CHECK-HTML-CIRCLE %s

// CHECK-JSON-INDEX: var JsonIndex = `
// CHECK-JSON-INDEX: {
// CHECK-JSON-INDEX:   "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX:   "Name": "",
// CHECK-JSON-INDEX:   "RefType": "default",
// CHECK-JSON-INDEX:   "Path": "",
// CHECK-JSON-INDEX:   "Children": [
// CHECK-JSON-INDEX:     {
// CHECK-JSON-INDEX:       "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX:       "Name": "GlobalNamespace",
// CHECK-JSON-INDEX:       "RefType": "namespace",
// CHECK-JSON-INDEX:       "Path": "GlobalNamespace",
// CHECK-JSON-INDEX:       "Children": [
// CHECK-JSON-INDEX:         {
// CHECK-JSON-INDEX:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX:           "Name": "Calculator",
// CHECK-JSON-INDEX:           "RefType": "record",
// CHECK-JSON-INDEX:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX:           "Children": []
// CHECK-JSON-INDEX:         },
// CHECK-JSON-INDEX:         {
// CHECK-JSON-INDEX:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX:           "Name": "Circle",
// CHECK-JSON-INDEX:           "RefType": "record",
// CHECK-JSON-INDEX:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX:           "Children": []
// CHECK-JSON-INDEX:         },
// CHECK-JSON-INDEX:         {
// CHECK-JSON-INDEX:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX:           "Name": "Rectangle",
// CHECK-JSON-INDEX:           "RefType": "record",
// CHECK-JSON-INDEX:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX:           "Children": []
// CHECK-JSON-INDEX:         },
// CHECK-JSON-INDEX:         {
// CHECK-JSON-INDEX:           "USR": "{{([0-9A-F]{40})}}",
// CHECK-JSON-INDEX:           "Name": "Shape",
// CHECK-JSON-INDEX:           "RefType": "record",
// CHECK-JSON-INDEX:           "Path": "GlobalNamespace",
// CHECK-JSON-INDEX:           "Children": []
// CHECK-JSON-INDEX:         }
// CHECK-JSON-INDEX:       ]
// CHECK-JSON-INDEX:     }
// CHECK-JSON-INDEX:   ]
// CHECK-JSON-INDEX: }`;

// CHECK-HTML-SHAPE: <!DOCTYPE html>
// CHECK-HTML-SHAPE: <meta charset="utf-8"/>
// CHECK-HTML-SHAPE: <title>class Shape</title>
// CHECK-HTML-SHAPE: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-SHAPE: <script src="{{.*}}index.js"></script>
// CHECK-HTML-SHAPE: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-SHAPE: <header id="project-title"></header>
// CHECK-HTML-SHAPE: <main>
// CHECK-HTML-SHAPE:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-SHAPE:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-SHAPE:     <h1>class Shape</h1>
// CHECK-HTML-SHAPE:     <p>Defined at line 8 of file {{.*}}Shape.h</p>
// CHECK-HTML-SHAPE:     <div>
// CHECK-HTML-SHAPE:       <div>
// CHECK-HTML-SHAPE:         <p> Provides a common interface for different types of shapes.</p>
// CHECK-HTML-SHAPE:       </div>
// CHECK-HTML-SHAPE:     </div>
// CHECK-HTML-SHAPE:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-SHAPE:     <div>
// CHECK-HTML-SHAPE:       <h3 id="{{([0-9A-F]{40})}}">~Shape</h3>
// CHECK-HTML-SHAPE:       <p>public void ~Shape()</p>
// CHECK-HTML-SHAPE:       <p>Defined at line 13 of file {{.*}}Shape.h</p>
// CHECK-HTML-SHAPE:       <div>
// CHECK-HTML-SHAPE:         <div></div>
// CHECK-HTML-SHAPE:       </div>
// CHECK-HTML-SHAPE:       <h3 id="{{([0-9A-F]{40})}}">area</h3>
// CHECK-HTML-SHAPE:       <p>public double area()</p>
// CHECK-HTML-SHAPE:       <div>
// CHECK-HTML-SHAPE:         <div></div>
// CHECK-HTML-SHAPE:       </div>
// CHECK-HTML-SHAPE:       <h3 id="{{([0-9A-F]{40})}}">perimeter</h3>
// CHECK-HTML-SHAPE:       <p>public double perimeter()</p>
// CHECK-HTML-SHAPE:       <div>
// CHECK-HTML-SHAPE:         <div></div>
// CHECK-HTML-SHAPE:       </div>
// CHECK-HTML-SHAPE:     </div>
// CHECK-HTML-SHAPE:   </div>
// CHECK-HTML-SHAPE:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-SHAPE:     <ol>
// CHECK-HTML-SHAPE:       <li>
// CHECK-HTML-SHAPE:         <span>
// CHECK-HTML-SHAPE:           <a href="#Functions">Functions</a>
// CHECK-HTML-SHAPE:         </span>
// CHECK-HTML-SHAPE:         <ul>
// CHECK-HTML-SHAPE:           <li>
// CHECK-HTML-SHAPE:             <span>
// CHECK-HTML-SHAPE:               <a href="#{{([0-9A-F]{40})}}">~Shape</a>
// CHECK-HTML-SHAPE:             </span>
// CHECK-HTML-SHAPE:           </li>
// CHECK-HTML-SHAPE:           <li>
// CHECK-HTML-SHAPE:             <span>
// CHECK-HTML-SHAPE:               <a href="#{{([0-9A-F]{40})}}">area</a>
// CHECK-HTML-SHAPE:             </span>
// CHECK-HTML-SHAPE:           </li>
// CHECK-HTML-SHAPE:           <li>
// CHECK-HTML-SHAPE:             <span>
// CHECK-HTML-SHAPE:               <a href="#{{([0-9A-F]{40})}}">perimeter</a>
// CHECK-HTML-SHAPE:             </span>
// CHECK-HTML-SHAPE:           </li>
// CHECK-HTML-SHAPE:         </ul>
// CHECK-HTML-SHAPE:       </li>
// CHECK-HTML-SHAPE:     </ol>
// CHECK-HTML-SHAPE:   </div>
// CHECK-HTML-SHAPE: </main>

// CHECK-HTML-CALC: <!DOCTYPE html>
// CHECK-HTML-CALC: <meta charset="utf-8"/>
// CHECK-HTML-CALC: <title>class Calculator</title>
// CHECK-HTML-CALC: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-CALC: <script src="{{.*}}index.js"></script>
// CHECK-HTML-CALC: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-CALC: <header id="project-title"></header>
// CHECK-HTML-CALC: <main>
// CHECK-HTML-CALC:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-CALC:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-CALC:     <h1>class Calculator</h1>
// CHECK-HTML-CALC:     <p>Defined at line 8 of file {{.*}}Calculator.h</p>
// CHECK-HTML-CALC:     <div>
// CHECK-HTML-CALC:       <div>
// CHECK-HTML-CALC:         <p> Provides basic arithmetic operations.</p>
// CHECK-HTML-CALC:       </div>
// CHECK-HTML-CALC:     </div>
// CHECK-HTML-CALC:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-CALC:     <div>
// CHECK-HTML-CALC:       <h3 id="{{([0-9A-F]{40})}}">add</h3>
// CHECK-HTML-CALC:       <p>public int add(int a, int b)</p>
// CHECK-HTML-CALC:       <p>Defined at line 4 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC:       <div>
// CHECK-HTML-CALC:         <div></div>
// CHECK-HTML-CALC:       </div>
// CHECK-HTML-CALC:       <h3 id="{{([0-9A-F]{40})}}">subtract</h3>
// CHECK-HTML-CALC:       <p>public int subtract(int a, int b)</p>
// CHECK-HTML-CALC:       <p>Defined at line 8 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC:       <div>
// CHECK-HTML-CALC:         <div></div>
// CHECK-HTML-CALC:       </div>
// CHECK-HTML-CALC:       <h3 id="{{([0-9A-F]{40})}}">multiply</h3>
// CHECK-HTML-CALC:       <p>public int multiply(int a, int b)</p>
// CHECK-HTML-CALC:       <p>Defined at line 12 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC:       <div>
// CHECK-HTML-CALC:         <div></div>
// CHECK-HTML-CALC:       </div>
// CHECK-HTML-CALC:       <h3 id="{{([0-9A-F]{40})}}">divide</h3>
// CHECK-HTML-CALC:       <p>public double divide(int a, int b)</p>
// CHECK-HTML-CALC:       <p>Defined at line 16 of file {{.*}}Calculator.cpp</p>
// CHECK-HTML-CALC:       <div>
// CHECK-HTML-CALC:         <div></div>
// CHECK-HTML-CALC:       </div>
// CHECK-HTML-CALC:     </div>
// CHECK-HTML-CALC:   </div>
// CHECK-HTML-CALC:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-CALC:     <ol>
// CHECK-HTML-CALC:       <li>
// CHECK-HTML-CALC:         <span>
// CHECK-HTML-CALC:           <a href="#Functions">Functions</a>
// CHECK-HTML-CALC:         </span>
// CHECK-HTML-CALC:         <ul>
// CHECK-HTML-CALC:           <li>
// CHECK-HTML-CALC:             <span>
// CHECK-HTML-CALC:               <a href="#{{([0-9A-F]{40})}}">add</a>
// CHECK-HTML-CALC:             </span>
// CHECK-HTML-CALC:           </li>
// CHECK-HTML-CALC:           <li>
// CHECK-HTML-CALC:             <span>
// CHECK-HTML-CALC:               <a href="#{{([0-9A-F]{40})}}">subtract</a>
// CHECK-HTML-CALC:             </span>
// CHECK-HTML-CALC:           </li>
// CHECK-HTML-CALC:           <li>
// CHECK-HTML-CALC:             <span>
// CHECK-HTML-CALC:               <a href="#{{([0-9A-F]{40})}}">multiply</a>
// CHECK-HTML-CALC:             </span>
// CHECK-HTML-CALC:           </li>
// CHECK-HTML-CALC:           <li>
// CHECK-HTML-CALC:             <span>
// CHECK-HTML-CALC:               <a href="#{{([0-9A-F]{40})}}">divide</a>
// CHECK-HTML-CALC:             </span>
// CHECK-HTML-CALC:           </li>
// CHECK-HTML-CALC:         </ul>
// CHECK-HTML-CALC:       </li>
// CHECK-HTML-CALC:     </ol>
// CHECK-HTML-CALC:   </div>
// CHECK-HTML-CALC: </main>

// CHECK-HTML-RECTANGLE: <!DOCTYPE html>
// CHECK-HTML-RECTANGLE: <meta charset="utf-8"/>
// CHECK-HTML-RECTANGLE: <title>class Rectangle</title>
// CHECK-HTML-RECTANGLE: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-RECTANGLE: <script src="{{.*}}index.js"></script>
// CHECK-HTML-RECTANGLE: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-RECTANGLE: <header id="project-title"></header>
// CHECK-HTML-RECTANGLE: <main>
// CHECK-HTML-RECTANGLE:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-RECTANGLE:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-RECTANGLE:     <h1>class Rectangle</h1>
// CHECK-HTML-RECTANGLE:     <p>Defined at line 10 of file {{.*}}Rectangle.h</p>
// CHECK-HTML-RECTANGLE:     <div>
// CHECK-HTML-RECTANGLE:       <div>
// CHECK-HTML-RECTANGLE:         <p> Represents a rectangle with a given width and height.</p>
// CHECK-HTML-RECTANGLE:       </div>
// CHECK-HTML-RECTANGLE:     </div>
// CHECK-HTML-RECTANGLE:     <p>
// CHECK-HTML-RECTANGLE:       Inherits from
// CHECK-HTML-RECTANGLE:       <a href="Shape.html">Shape</a>
// CHECK-HTML-RECTANGLE:     </p>
// CHECK-HTML-RECTANGLE:     <h2 id="Members">Members</h2>
// CHECK-HTML-RECTANGLE:     <ul>
// CHECK-HTML-RECTANGLE:       <li>private double width_</li>
// CHECK-HTML-RECTANGLE:       <li>private double height_</li>
// CHECK-HTML-RECTANGLE:     </ul>
// CHECK-HTML-RECTANGLE:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-RECTANGLE:     <div>
// CHECK-HTML-RECTANGLE:       <h3 id="{{([0-9A-F]{40})}}">Rectangle</h3>
// CHECK-HTML-RECTANGLE:       <p>public void Rectangle(double width, double height)</p>
// CHECK-HTML-RECTANGLE:       <p>Defined at line 3 of file {{.*}}Rectangle.cpp</p>
// CHECK-HTML-RECTANGLE:       <div>
// CHECK-HTML-RECTANGLE:         <div></div>
// CHECK-HTML-RECTANGLE:       </div>
// CHECK-HTML-RECTANGLE:       <h3 id="{{([0-9A-F]{40})}}">area</h3>
// CHECK-HTML-RECTANGLE:       <p>public double area()</p>
// CHECK-HTML-RECTANGLE:       <p>Defined at line 6 of file {{.*}}Rectangle.cpp</p>
// CHECK-HTML-RECTANGLE:       <div>
// CHECK-HTML-RECTANGLE:         <div></div>
// CHECK-HTML-RECTANGLE:       </div>
// CHECK-HTML-RECTANGLE:       <h3 id="{{([0-9A-F]{40})}}">perimeter</h3>
// CHECK-HTML-RECTANGLE:       <p>public double perimeter()</p>
// CHECK-HTML-RECTANGLE:       <p>Defined at line 10 of file {{.*}}Rectangle.cpp</p>
// CHECK-HTML-RECTANGLE:       <div>
// CHECK-HTML-RECTANGLE:         <div></div>
// CHECK-HTML-RECTANGLE:       </div>
// CHECK-HTML-RECTANGLE:     </div>
// CHECK-HTML-RECTANGLE:   </div>
// CHECK-HTML-RECTANGLE:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-RECTANGLE:     <ol>
// CHECK-HTML-RECTANGLE:       <li>
// CHECK-HTML-RECTANGLE:         <span>
// CHECK-HTML-RECTANGLE:           <a href="#Members">Members</a>
// CHECK-HTML-RECTANGLE:         </span>
// CHECK-HTML-RECTANGLE:       </li>
// CHECK-HTML-RECTANGLE:       <li>
// CHECK-HTML-RECTANGLE:         <span>
// CHECK-HTML-RECTANGLE:           <a href="#Functions">Functions</a>
// CHECK-HTML-RECTANGLE:         </span>
// CHECK-HTML-RECTANGLE:         <ul>
// CHECK-HTML-RECTANGLE:           <li>
// CHECK-HTML-RECTANGLE:             <span>
// CHECK-HTML-RECTANGLE:               <a href="#{{([0-9A-F]{40})}}">Rectangle</a>
// CHECK-HTML-RECTANGLE:             </span>
// CHECK-HTML-RECTANGLE:           </li>
// CHECK-HTML-RECTANGLE:           <li>
// CHECK-HTML-RECTANGLE:             <span>
// CHECK-HTML-RECTANGLE:               <a href="#{{([0-9A-F]{40})}}">area</a>
// CHECK-HTML-RECTANGLE:             </span>
// CHECK-HTML-RECTANGLE:           </li>
// CHECK-HTML-RECTANGLE:           <li>
// CHECK-HTML-RECTANGLE:             <span>
// CHECK-HTML-RECTANGLE:               <a href="#{{([0-9A-F]{40})}}">perimeter</a>
// CHECK-HTML-RECTANGLE:             </span>
// CHECK-HTML-RECTANGLE:           </li>
// CHECK-HTML-RECTANGLE:         </ul>
// CHECK-HTML-RECTANGLE:       </li>
// CHECK-HTML-RECTANGLE:     </ol>
// CHECK-HTML-RECTANGLE:   </div>
// CHECK-HTML-RECTANGLE: </main>

// CHECK-HTML-CIRCLE: <!DOCTYPE html>
// CHECK-HTML-CIRCLE: <meta charset="utf-8"/>
// CHECK-HTML-CIRCLE: <title>class Circle</title>
// CHECK-HTML-CIRCLE: <link rel="stylesheet" href="{{.*}}clang-doc-default-stylesheet.css"/>
// CHECK-HTML-CIRCLE: <script src="{{.*}}index.js"></script>
// CHECK-HTML-CIRCLE: <script src="{{.*}}index_json.js"></script>
// CHECK-HTML-CIRCLE: <header id="project-title"></header>
// CHECK-HTML-CIRCLE: <main>
// CHECK-HTML-CIRCLE:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// CHECK-HTML-CIRCLE:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// CHECK-HTML-CIRCLE:     <h1>class Circle</h1>
// CHECK-HTML-CIRCLE:     <p>Defined at line 10 of file {{.*}}Circle.h</p>
// CHECK-HTML-CIRCLE:     <div>
// CHECK-HTML-CIRCLE:       <div>
// CHECK-HTML-CIRCLE:         <p> Represents a circle with a given radius.</p>
// CHECK-HTML-CIRCLE:       </div>
// CHECK-HTML-CIRCLE:     </div>
// CHECK-HTML-CIRCLE:     <p>
// CHECK-HTML-CIRCLE:       Inherits from
// CHECK-HTML-CIRCLE:       <a href="Shape.html">Shape</a>
// CHECK-HTML-CIRCLE:     </p>
// CHECK-HTML-CIRCLE:     <h2 id="Members">Members</h2>
// CHECK-HTML-CIRCLE:     <ul>
// CHECK-HTML-CIRCLE:       <li>private double radius_</li>
// CHECK-HTML-CIRCLE:     </ul>
// CHECK-HTML-CIRCLE:     <h2 id="Functions">Functions</h2>
// CHECK-HTML-CIRCLE:     <div>
// CHECK-HTML-CIRCLE:       <h3 id="{{([0-9A-F]{40})}}">Circle</h3>
// CHECK-HTML-CIRCLE:       <p>public void Circle(double radius)</p>
// CHECK-HTML-CIRCLE:       <p>Defined at line 3 of file {{.*}}Circle.cpp</p>
// CHECK-HTML-CIRCLE:       <div>
// CHECK-HTML-CIRCLE:         <div></div>
// CHECK-HTML-CIRCLE:       </div>
// CHECK-HTML-CIRCLE:       <h3 id="{{([0-9A-F]{40})}}">area</h3>
// CHECK-HTML-CIRCLE:       <p>public double area()</p>
// CHECK-HTML-CIRCLE:       <p>Defined at line 5 of file {{.*}}Circle.cpp</p>
// CHECK-HTML-CIRCLE:       <div>
// CHECK-HTML-CIRCLE:         <div></div>
// CHECK-HTML-CIRCLE:       </div>
// CHECK-HTML-CIRCLE:       <h3 id="{{([0-9A-F]{40})}}">perimeter</h3>
// CHECK-HTML-CIRCLE:       <p>public double perimeter()</p>
// CHECK-HTML-CIRCLE:       <p>Defined at line 9 of file {{.*}}Circle.cpp</p>
// CHECK-HTML-CIRCLE:       <div>
// CHECK-HTML-CIRCLE:         <div></div>
// CHECK-HTML-CIRCLE:       </div>
// CHECK-HTML-CIRCLE:     </div>
// CHECK-HTML-CIRCLE:   </div>
// CHECK-HTML-CIRCLE:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// CHECK-HTML-CIRCLE:     <ol>
// CHECK-HTML-CIRCLE:       <li>
// CHECK-HTML-CIRCLE:         <span>
// CHECK-HTML-CIRCLE:           <a href="#Members">Members</a>
// CHECK-HTML-CIRCLE:         </span>
// CHECK-HTML-CIRCLE:       </li>
// CHECK-HTML-CIRCLE:       <li>
// CHECK-HTML-CIRCLE:         <span>
// CHECK-HTML-CIRCLE:           <a href="#Functions">Functions</a>
// CHECK-HTML-CIRCLE:         </span>
// CHECK-HTML-CIRCLE:         <ul>
// CHECK-HTML-CIRCLE:           <li>
// CHECK-HTML-CIRCLE:             <span>
// CHECK-HTML-CIRCLE:               <a href="#{{([0-9A-F]{40})}}">Circle</a>
// CHECK-HTML-CIRCLE:             </span>
// CHECK-HTML-CIRCLE:           </li>
// CHECK-HTML-CIRCLE:           <li>
// CHECK-HTML-CIRCLE:             <span>
// CHECK-HTML-CIRCLE:               <a href="#{{([0-9A-F]{40})}}">area</a>
// CHECK-HTML-CIRCLE:             </span>
// CHECK-HTML-CIRCLE:           </li>
// CHECK-HTML-CIRCLE:           <li>
// CHECK-HTML-CIRCLE:             <span>
// CHECK-HTML-CIRCLE:               <a href="#{{([0-9A-F]{40})}}">perimeter</a>
// CHECK-HTML-CIRCLE:             </span>
// CHECK-HTML-CIRCLE:           </li>
// CHECK-HTML-CIRCLE:         </ul>
// CHECK-HTML-CIRCLE:       </li>
// CHECK-HTML-CIRCLE:     </ol>
// CHECK-HTML-CIRCLE:   </div>
// CHECK-HTML-CIRCLE: </main>






