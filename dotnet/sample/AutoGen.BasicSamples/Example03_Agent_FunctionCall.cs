﻿// Copyright (c) 2023 - 2024, Owners of https://github.com/ag2labs
// SPDX-License-Identifier: Apache-2.0
// Contributions to this project, i.e., https://github.com/ag2labs/ag2, 
// are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Portions derived from  https://github.com/microsoft/autogen under the MIT License.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// Example03_Agent_FunctionCall.cs

using AutoGen;
using AutoGen.BasicSample;
using AutoGen.Core;
using FluentAssertions;

/// <summary>
/// This example shows how to add type-safe function call to an agent.
/// </summary>
public partial class Example03_Agent_FunctionCall
{
    /// <summary>
    /// upper case the message when asked.
    /// </summary>
    /// <param name="message"></param>
    [Function]
    public async Task<string> UpperCase(string message)
    {
        return message.ToUpper();
    }

    /// <summary>
    /// Concatenate strings.
    /// </summary>
    /// <param name="strings">strings to concatenate</param>
    [Function]
    public async Task<string> ConcatString(string[] strings)
    {
        return string.Join(" ", strings);
    }

    /// <summary>
    /// calculate tax
    /// </summary>
    /// <param name="price">price, should be an integer</param>
    /// <param name="taxRate">tax rate, should be in range (0, 1)</param>
    [FunctionAttribute]
    public async Task<string> CalculateTax(int price, float taxRate)
    {
        return $"tax is {price * taxRate}";
    }

    public static async Task RunAsync()
    {
        var instance = new Example03_Agent_FunctionCall();
        var gpt35 = LLMConfiguration.GetAzureOpenAIGPT3_5_Turbo();

        // AutoGen makes use of AutoGen.SourceGenerator to automatically generate FunctionDefinition and FunctionCallWrapper for you.
        // The FunctionDefinition will be created based on function signature and XML documentation.
        // The return type of type-safe function needs to be Task<string>. And to get the best performance, please try only use primitive types and arrays of primitive types as parameters.
        var config = new ConversableAgentConfig
        {
            Temperature = 0,
            ConfigList = [gpt35],
            FunctionContracts = new[]
            {
                instance.ConcatStringFunctionContract,
                instance.UpperCaseFunctionContract,
                instance.CalculateTaxFunctionContract,
            },
        };

        var agent = new AssistantAgent(
            name: "agent",
            systemMessage: "You are a helpful AI assistant",
            llmConfig: config,
            functionMap: new Dictionary<string, Func<string, Task<string>>>
            {
                { nameof(ConcatString), instance.ConcatStringWrapper },
                { nameof(UpperCase), instance.UpperCaseWrapper },
                { nameof(CalculateTax), instance.CalculateTaxWrapper },
            })
            .RegisterPrintMessage();

        // talk to the assistant agent
        var upperCase = await agent.SendAsync("convert to upper case: hello world");
        upperCase.GetContent()?.Should().Be("HELLO WORLD");
        upperCase.Should().BeOfType<ToolCallAggregateMessage>();
        upperCase.GetToolCalls().Should().HaveCount(1);
        upperCase.GetToolCalls().First().FunctionName.Should().Be(nameof(UpperCase));

        var concatString = await agent.SendAsync("concatenate strings: a, b, c, d, e");
        concatString.GetContent()?.Should().Be("a b c d e");
        concatString.Should().BeOfType<ToolCallAggregateMessage>();
        concatString.GetToolCalls().Should().HaveCount(1);
        concatString.GetToolCalls().First().FunctionName.Should().Be(nameof(ConcatString));

        var calculateTax = await agent.SendAsync("calculate tax: 100, 0.1");
        calculateTax.GetContent().Should().Be("tax is 10");
        calculateTax.Should().BeOfType<ToolCallAggregateMessage>();
        calculateTax.GetToolCalls().Should().HaveCount(1);
        calculateTax.GetToolCalls().First().FunctionName.Should().Be(nameof(CalculateTax));

        // parallel function calls
        var calculateTaxes = await agent.SendAsync("calculate tax: 100, 0.1; calculate tax: 200, 0.2");
        calculateTaxes.GetContent().Should().Be("tax is 10\ntax is 40"); // "tax is 10\n tax is 40
        calculateTaxes.Should().BeOfType<ToolCallAggregateMessage>();
        calculateTaxes.GetToolCalls().Should().HaveCount(2);
        calculateTaxes.GetToolCalls().First().FunctionName.Should().Be(nameof(CalculateTax));

        // send aggregate message back to llm to get the final result
        var finalResult = await agent.SendAsync(calculateTaxes);
    }
}
