#pragma once

#include <CLI11.hpp>

struct Flag {
        Flag(bool def) : def(def) {}
        CLI::Option* on = nullptr;
        CLI::Option* off = nullptr;
        bool def;
        void add_to(CLI::App& app, std::string name_on, std::string name_off) {
                on = app.add_flag(name_on);
                off = app.add_flag(name_off);
                on->excludes(off);
                off->excludes(on);
                if (def) {
                        on->default_val("TRUE");
                } else {
                        off->default_val("TRUE");
                }
        }
        bool get() const {
                if (*on) return true;
                if (*off) return false;
                return def;
        }
        operator bool() const { return get(); }
};
