﻿// Copyright (c) 2023 - 2024, Owners of https://github.com/ag2labs
// SPDX-License-Identifier: Apache-2.0
// Contributions to this project, i.e., https://github.com/ag2labs/ag2, 
// are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Portions derived from  https://github.com/microsoft/autogen under the MIT License.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// DotnetInteractiveServiceTest.cs

using FluentAssertions;
using Xunit;
using Xunit.Abstractions;

namespace AutoGen.DotnetInteractive.Tests;

[Collection("Sequential")]
public class DotnetInteractiveServiceTest : IDisposable
{
    private ITestOutputHelper _output;
    private InteractiveService _interactiveService;
    private string _workingDir;

    public DotnetInteractiveServiceTest(ITestOutputHelper output)
    {
        _output = output;
        _workingDir = Path.Combine(Path.GetTempPath(), "test", Path.GetRandomFileName());
        if (!Directory.Exists(_workingDir))
        {
            Directory.CreateDirectory(_workingDir);
        }

        _interactiveService = new InteractiveService(_workingDir);
        _interactiveService.StartAsync(_workingDir, default).Wait();
    }

    public void Dispose()
    {
        _interactiveService.Dispose();
    }

    [Fact]
    public async Task ItRunCSharpCodeSnippetTestsAsync()
    {
        var cts = new CancellationTokenSource();
        var isRunning = await _interactiveService.StartAsync(_workingDir, cts.Token);

        isRunning.Should().BeTrue();

        _interactiveService.IsRunning().Should().BeTrue();

        // test code snippet
        var hello_world = @"
Console.WriteLine(""hello world"");
";

        await this.TestCSharpCodeSnippet(_interactiveService, hello_world, "hello world");
        await this.TestCSharpCodeSnippet(
            _interactiveService,
            code: @"
Console.WriteLine(""hello world""
",
            expectedOutput: "Error: (2,32): error CS1026: ) expected");

        await this.TestCSharpCodeSnippet(
            service: _interactiveService,
            code: "throw new Exception();",
            expectedOutput: "Error: System.Exception: Exception of type 'System.Exception' was thrown");
    }

    [Fact]
    public async Task ItRunPowershellScriptTestsAsync()
    {
        // test power shell
        var ps = @"Write-Output ""hello world""";
        await this.TestPowershellCodeSnippet(_interactiveService, ps, "hello world");
    }

    private async Task TestPowershellCodeSnippet(InteractiveService service, string code, string expectedOutput)
    {
        var result = await service.SubmitPowershellCodeAsync(code, CancellationToken.None);
        result.Should().StartWith(expectedOutput);
    }

    private async Task TestCSharpCodeSnippet(InteractiveService service, string code, string expectedOutput)
    {
        var result = await service.SubmitCSharpCodeAsync(code, CancellationToken.None);
        result.Should().StartWith(expectedOutput);
    }
}
