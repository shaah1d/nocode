"use client"
import React, { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import { TypingAnimation } from "../magicui/typing-animation";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import Navbar from "../Layout/Navbar";


const formSchema = z.object({
  question: z.string().min(2, {
    message: "Question must be at least 2 characters.",
  }),
});

const LandingPage = () => {

  const [answer, setAnswer] = useState("");
  const [qasked, setQasked] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoad, setIsLoad] = useState(false);
  const [showForm, setShowForm] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      question: "",
    },
  });

  const onSubmit = async (data: z.infer<typeof formSchema>) => {
    try {
      setIsLoading(true);
      const response = await fetch("/api/suggest", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const result = await response.json();
      setAnswer(result.answer);
      setQasked(true);
      setShowForm(false);
    } catch (error) {
      console.error("Error submitting form:", error);
      alert("An error occurred while submitting the form.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f8f8]">
   <Navbar />
      <main className="relative px-6 pt-12">
        <div
          className="absolute right-0 top-0 h-[300px] w-[300px] animate-pulse rounded-full bg-gradient-to-br from-pink-400 via-orange-300 to-yellow-200 opacity-70 blur-3xl"
          aria-hidden="true"
        />
        <div className="relative flex flex-col md:flex-row justify-between items-center">
          {/* Left Section */}

          <div className="w-full md:w-1/2 max-w-md">
            <h1 className="max-w-3xl text-5xl font-light leading-tight tracking-tight mb-4">
              Make Deployable AI
              <br />
              Models in <span className="bg-yellow-700 text-[#f8f8f8] pr-2 pl-3 line-through">days</span>
              <br />
              minutes.
            </h1>
            <div className="mt-24">
            {!showForm ? (
              <Button
                variant="outline"
                className="rounded-full border-2 px-8"
                onClick={() => setShowForm(true)}
              >
                <span className="relative">
                  TRY OUT WITH A PROMPT
                  <div className="absolute -left-4 -right-4 -top-4 -bottom-4 animate-spin-slow rounded-full border border-black opacity-50"></div>
                </span>
              </Button>
            ) : (
              <div className="w-full max-w-md">
                {!qasked ? (
                  <Form {...form}>
                    <form
                      onSubmit={form.handleSubmit(onSubmit)}
                      className="flex gap-2"
                    >
                      <FormField
                        control={form.control}
                        name="question"
                        render={({ field }) => (
                          <FormItem className="flex-grow">
                            <FormControl>
                              <Input
                                placeholder="Write the topic here"
                                {...field}
                                disabled={isLoading || isLoad}
                                className="border-none focus:ring-0 focus-visible:ring-0 focus-visible:ring-offset-0"
                                autoComplete="off"
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <Button
                        type="submit"
                        disabled={isLoading || isLoad}
                        className="bg-grey- rounded-full bg-yellow-700 hover:bg-yellow-800 h-8 w-8 p-0 flex items-center justify-center"
                      >
                        {isLoading || isLoad ? (
                          <span className="h-4 w-4">...</span>
                        ) : (
                          <Send className="h-4 w-4" />
                        )}
                      </Button>
                    </form>
                  </Form>
                ) : null}
              </div>
            )}
            </div>
            <p className="mt-8 text-sm leading-relaxed text-gray-600">
              Design, train, and deploy powerful AI models effortlessly using a
              fully visual, no-code interface.
              <br />
              Customize workflows, integrate data, and bring your AI ideas to
              life—no programming skills required!
            </p>
          </div>

          {/* Right Section (Answer Block) */}
          {qasked && (
            <div className="w-full md:w-1/2 p-6 rounded-lg mt-8 md:mt-0">
              <TypingAnimation>{answer}</TypingAnimation>
              {/* <pre className="whitespace-pre-wrap text-lg"></pre> */}
              <div className="flex gap-4 mt-4">
          
                <Button className="bg-yellow-700 hover:bg-yellow-800"><a href="/playground">Try our playground &rarr;</a></Button>


              </div>
            </div>
          )}
        </div>
        
      </main>
    </div>
  );
};

export default LandingPage;