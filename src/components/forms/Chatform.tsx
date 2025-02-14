"use client";

import React, { useState } from 'react';
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";

const formSchema = z.object({
  question: z.string().min(2, {
    message: "Question must be at least 2 characters.",
  }),
});

export default function LandingPage() {
  const router = useRouter();
  const [answer, setAnswer] = useState("");
  const [qasked, setQasked] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoad, setIsLoad] = useState(false);

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
    } catch (error) {
      console.error("Error submitting form:", error);
      alert("An error occurred while submitting the form.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid sm:grid-cols-12 h-screen">
      <div className="sm:col-span-7 flex justify-center items-center h-full">
        <div className="p-10">
          <h1 className="text-4xl font-bold">NoCode AI Model Builder – Drag, Drop, and Deploy AI Without Coding</h1>
          <p className="text-lg mt-3">
          Design, train, and deploy powerful AI models effortlessly using a fully visual, no-code interface. Customize workflows, integrate data, and bring your AI ideas to life—no programming skills required!
          </p>
          <p className="text-md mt-2 mb-2 font-bold">Try it out!</p>
          
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="flex">
              <FormField
                control={form.control}
                name="question"
                render={({ field }) => (
                  <FormItem className="w-[80%]">
                    <FormControl>
                      <Input
                        placeholder="Write the topic here"
                        {...field}
                        disabled={isLoading || isLoad}
                        className="input input-bordered min-w-sm"
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button
                type="submit"
                className="btn btn-primary ml-4"
                disabled={isLoading || isLoad}
              >
                {isLoading || isLoad ? 'Loading...' : 'Give it a shot'}
              </Button>
            </form>
          </Form>

         
        </div>
      </div>
      <div className="sm:col-span-5 h-full overflow-hidden">
        {!qasked ? (
          <img src="/landing.svg" alt="Quiz design" className=" w-full h-full" />
        ) : (
          <div className="h-full flex flex-col p-6 bg-gray-50">
            <pre className="whitespace-pre-wrap flex-grow overflow-y-auto text-md">{answer}</pre>
            <div className="flex gap-4 mt-4">
              <Button
                onClick={() => {
                  router.push('/playground');
                }}
              >
                Try our playground &rarr;
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setQasked(false);
                  setAnswer("");
                  form.reset();
                }}
              >
                Try another topic
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}