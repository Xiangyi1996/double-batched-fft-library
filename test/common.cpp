// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"

#include "common_host.h"

TEST_CASE("tinydpcppnn::format empty args") {

    std::string input{"This is a teststring which should work fine"};
    std::string output = tinydpcppnn::format(input);

    CHECK(input == output);
}

TEST_CASE("tinydpcppnn::format 1 string arg") {

    std::string input{"This is a teststring which should {} work fine"};
    std::string output = tinydpcppnn::format(input, "hopefully");

    CHECK(output == "This is a teststring which should hopefully work fine");
}

TEST_CASE("tinydpcppnn::format 1 int arg") {

    std::string input{"This is a teststring which should {} work fine"};
    std::string output = tinydpcppnn::format(input, 5);

    CHECK(output == "This is a teststring which should 5 work fine");
}

TEST_CASE("tinydpcppnn::format 1 double arg") {

    std::string input{"This is a teststring which should {} work fine"};
    std::string output = tinydpcppnn::format(input, 5.1);

    CHECK(output == "This is a teststring which should 5.1 work fine");
}

TEST_CASE("tinydpcppnn::format 0 arg 1 bracket") {

    std::string input{"This is a teststring which should {} work fine"};
    CHECK(tinydpcppnn::format(input) == input);
}

TEST_CASE("tinydpcppnn::format 2 arg 1 bracket") {

    std::string input{"This is a teststring which should {} work fine"};
    CHECK_THROWS_AS(tinydpcppnn::format(input, 1, 2), std::invalid_argument);
}

TEST_CASE("tinydpcppnn::format 2 arg 2 brackets") {

    std::string input{"This is a teststring which should {} work fine {}"};
    std::string output = tinydpcppnn::format(input, 1, "!");
    CHECK(output == "This is a teststring which should 1 work fine !");
}

TEST_CASE("tinydpcppnn::format 1 arg 1 incomplete bracket") {

    std::string input{"This is a teststring which should not { work fine"};
    CHECK_THROWS_AS(tinydpcppnn::format(input, 1), std::invalid_argument);
}

TEST_CASE("isequalstring 1") { CHECK(isequalstring("TEST", "TEST")); }
TEST_CASE("isequalstring 2") { CHECK(isequalstring("TesT", "TEST")); }
TEST_CASE("isequalstring 3") { CHECK(!isequalstring("TESTE", "TEST")); }
TEST_CASE("isequalstring 4") { CHECK(!isequalstring("TEST", "TESTE")); }
TEST_CASE("isequalstring 5") { CHECK(!isequalstring("", "TESTE")); }
TEST_CASE("isequalstring 6") { CHECK(isequalstring("tEsT", "TESt")); }