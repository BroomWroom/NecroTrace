"use client";

import React from "react";
import { Twitter, Github, Twitch, Mail } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";

const footerLinks = [
  { title: "Overview", href: "#" },
  { title: "Features", href: "#" },
  { title: "Methodology", href: "#" },
  { title: "Dossier", href: "#" },
  { title: "Examination", href: "?view=examination" },
  { title: "Contact", href: "mailto:forensics@necrotrace.org" },
];

const Footer = () => {
  return (
    <footer className="w-full bg-black text-foreground">
      <div className="mx-auto max-w-6xl px-6 py-14">
        <div className="flex flex-col md:flex-row justify-between items-start gap-8 pb-10">
          {/* Logo & Links */}
          <div className="space-y-6">
            <div className="flex items-center gap-2.5">
              <img
                src="/assets/logo_icon.png"
                alt="NecroTrace Logo"
                width={32}
                height={32}
                className="h-8 w-8 object-contain drop-shadow-[0_0_8px_rgba(116,194,92,0.45)]"
              />
              <span className="text-xl font-bold tracking-tight text-white">necrotrace</span>
            </div>

            <ul className="flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
              {footerLinks.map(({ title, href }) => (
                <li key={title}>
                  <Link className="hover:text-white transition-colors" href={href}>
                    {title}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Stay up to date */}
          <div className="w-full max-w-sm space-y-3">
            <h6 className="text-sm font-semibold text-white">Stay up to date</h6>
            <form onSubmit={(e) => e.preventDefault()} className="flex items-center gap-2">
              <Input
                placeholder="Enter your email"
                type="email"
                className="h-10 bg-black border-zinc-800 text-sm text-white placeholder:text-zinc-500 focus-visible:ring-zinc-600"
              />
              <Button className="h-10 bg-white text-black hover:bg-zinc-200 font-medium px-5">
                Subscribe
              </Button>
            </form>
          </div>
        </div>

        <Separator className="bg-zinc-800" />

        <div className="flex flex-col-reverse items-center justify-between gap-4 pt-8 text-sm text-zinc-500 sm:flex-row">
          <span>&copy; {new Date().getFullYear()} NecroTrace. All rights reserved.</span>

          <div className="flex items-center gap-5 text-zinc-500">
            <Link href="mailto:forensics@necrotrace.org" className="hover:text-white transition-colors">
              <Mail className="h-5 w-5" />
            </Link>
            <Link href="https://twitter.com" target="_blank" rel="noreferrer" className="hover:text-white transition-colors">
              <Twitter className="h-5 w-5" />
            </Link>
            <Link href="https://twitch.tv" target="_blank" rel="noreferrer" className="hover:text-white transition-colors">
              <Twitch className="h-5 w-5" />
            </Link>
            <Link href="https://github.com/BroomWroom/NecroTrace" target="_blank" rel="noreferrer" className="hover:text-white transition-colors">
              <Github className="h-5 w-5" />
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
