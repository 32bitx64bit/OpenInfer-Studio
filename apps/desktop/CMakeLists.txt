cmake_minimum_required(VERSION 3.24)

# Single source of truth: internal/version/VERSION (shared with the Go backend).
file(READ "${CMAKE_CURRENT_LIST_DIR}/../../internal/version/VERSION" OPENINFER_VERSION_RAW)
string(STRIP "${OPENINFER_VERSION_RAW}" OPENINFER_VERSION)
if(NOT OPENINFER_VERSION MATCHES "^[0-9]+\\.[0-9]+")
    message(FATAL_ERROR "invalid OPENINFER_VERSION from internal/version/VERSION: '${OPENINFER_VERSION}'")
endif()

project(openinfer-studio VERSION ${OPENINFER_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick QuickControls2 WebSockets Widgets)

qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(openinfer-studio
    main.cpp
    resources.qrc
)

# AUTORCC sometimes only watches resources.qrc, not the QML it lists, so a
# QML-only edit would leave the old pages baked into the binary.
file(GLOB_RECURSE OPENINFER_QML_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/qml/*.qml"
    "${CMAKE_CURRENT_SOURCE_DIR}/qml/*.js"
)
set_property(SOURCE resources.qrc APPEND PROPERTY OBJECT_DEPENDS ${OPENINFER_QML_FILES})

target_compile_definitions(openinfer-studio PRIVATE
    OPENINFER_VERSION="${OPENINFER_VERSION}"
)

target_link_libraries(openinfer-studio PRIVATE
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::WebSockets
    Qt6::Widgets
)

set_target_properties(openinfer-studio PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_BUNDLE_NAME "OpenInfer Studio"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.openinfer.studio"
    MACOSX_BUNDLE_BUNDLE_VERSION "${OPENINFER_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${OPENINFER_VERSION}"
)

install(TARGETS openinfer-studio
    BUNDLE DESTINATION .
    RUNTIME DESTINATION bin
)
